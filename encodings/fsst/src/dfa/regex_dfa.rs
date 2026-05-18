// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! # General regex DFA over FSST symbols
//!
//! Compiles an arbitrary regular expression into a byte-level DFA via
//! [`regex_automata::dfa::dense`], then lifts the byte transitions into a
//! per-symbol transition table. The result evaluates `Regex::is_match`
//! semantics directly on FSST-compressed bytes without decompressing.
//!
//! ## Algorithm
//!
//! 1. **Build the byte DFA.** `dense::Builder` produces a deterministic
//!    automaton over bytes that models unanchored search semantics
//!    (`Anchored::No`) — the start state implicitly allows the match to
//!    begin at any position via a `.*?` prefix.
//!
//! 2. **BFS reachable states.** The DFA exposes states as opaque
//!    [`StateID`]s. We discover everything reachable from the start by
//!    BFS through all 256 byte transitions and the end-of-input
//!    transition, mapping each visited [`StateID`] to a compact `u32`
//!    index. Construction bails to `Ok(None)` if the state count exceeds
//!    [`RegexFsstDfa::MAX_STATES`] or if any reachable state is a quit
//!    state (regex_automata's signal that it gave up on a state).
//!
//! 3. **Build byte / EOI / match tables.** For each compact state we
//!    record its byte transitions, its end-of-input transition, and its
//!    `is_match` / `is_dead` flags.
//!
//! 4. **Lift to symbol transitions.** For every `(state, symbol)` we
//!    simulate the symbol's bytes through the byte table. If processing
//!    enters a match or dead state mid-symbol we early-break — once
//!    `is_match` we've already succeeded for the row, and once `is_dead`
//!    we've already failed, so processing further bytes is wasted work.
//!
//! ## Escape handling
//!
//! FSST reserves byte 255 ([`fsst::ESCAPE_CODE`]) to mean "the next byte
//! is a literal." The runtime [`matches`] loop treats escape bytes as a
//! signal to step the DFA with the next byte using the byte transition
//! table directly, rather than the symbol table.
//!
//! ## Limits
//!
//! We cap reachable states at [`MAX_STATES`] = 512 to keep the symbol
//! table bounded (≈512 KB for 256 symbols). Patterns whose minimal DFA
//! exceeds this limit fall back to the canonical execution path. Most
//! SQL regex workloads compile to a few tens of states.

use std::collections::VecDeque;

use fsst::ESCAPE_CODE;
use fsst::Symbol;
use regex_automata::Anchored;
use regex_automata::dfa::Automaton;
use regex_automata::dfa::dense;
use regex_automata::util::primitives::StateID;
use regex_automata::util::start;
use regex_automata::util::syntax;
use rustc_hash::FxHashMap;
use vortex_error::VortexResult;

/// A DFA that evaluates an arbitrary regular expression directly on
/// FSST-compressed code streams.
pub(crate) struct RegexFsstDfa {
    n_symbols: usize,
    /// `[state * n_symbols + code] -> next_state`
    symbol_trans: Vec<u32>,
    /// `[state * 256 + byte] -> next_state`, used for escaped literal bytes.
    byte_trans: Vec<u32>,
    /// `[state] -> next_state` after end-of-input.
    eoi_trans: Vec<u32>,
    is_match: Vec<bool>,
    is_dead: Vec<bool>,
    start: u32,
}

impl RegexFsstDfa {
    /// Upper bound on reachable byte-DFA states. Beyond this we give up
    /// and let the canonical (decompressing) path handle the pattern.
    pub(crate) const MAX_STATES: usize = 512;

    /// Compile `pattern` and lift it onto the supplied FSST symbol table.
    ///
    /// Returns:
    /// - `Ok(Some(dfa))` if the regex compiles and its reachable state
    ///   space fits within [`MAX_STATES`].
    /// - `Ok(None)` if the regex is unsupported by regex_automata, has
    ///   too many states, or reaches a quit state.
    /// - `Err(_)` only for hard internal errors (none today).
    pub(crate) fn try_new(
        symbols: &[Symbol],
        symbol_lengths: &[u8],
        pattern: &str,
        case_insensitive: bool,
    ) -> VortexResult<Option<Self>> {
        let syntax_cfg = syntax::Config::new().case_insensitive(case_insensitive);
        let dfa = match dense::Builder::new().syntax(syntax_cfg).build(pattern) {
            Ok(dfa) => dfa,
            Err(_) => return Ok(None),
        };

        let start_cfg = start::Config::new().anchored(Anchored::No);
        let regex_start = match dfa.start_state(&start_cfg) {
            Ok(s) => s,
            Err(_) => return Ok(None),
        };

        // BFS reachable states, mapping each opaque StateID to a compact u32.
        let mut state_map: FxHashMap<StateID, u32> = FxHashMap::default();
        let mut state_list: Vec<StateID> = Vec::new();
        let mut queue: VecDeque<StateID> = VecDeque::new();

        let register = |id: StateID,
                        state_map: &mut FxHashMap<StateID, u32>,
                        state_list: &mut Vec<StateID>,
                        queue: &mut VecDeque<StateID>|
         -> Result<(), ()> {
            if state_map.contains_key(&id) {
                return Ok(());
            }
            if state_list.len() >= Self::MAX_STATES {
                return Err(());
            }
            let compact = u32::try_from(state_list.len()).map_err(|_| ())?;
            state_map.insert(id, compact);
            state_list.push(id);
            queue.push_back(id);
            Ok(())
        };

        if register(regex_start, &mut state_map, &mut state_list, &mut queue).is_err() {
            return Ok(None);
        }

        while let Some(state) = queue.pop_front() {
            if dfa.is_quit_state(state) {
                return Ok(None);
            }
            for byte in 0..=255u8 {
                let next = dfa.next_state(state, byte);
                if register(next, &mut state_map, &mut state_list, &mut queue).is_err() {
                    return Ok(None);
                }
            }
            let next_eoi = dfa.next_eoi_state(state);
            if register(next_eoi, &mut state_map, &mut state_list, &mut queue).is_err() {
                return Ok(None);
            }
        }

        let n_states = state_list.len();
        let mut byte_trans = vec![0u32; n_states * 256];
        let mut eoi_trans = vec![0u32; n_states];
        let mut is_match = vec![false; n_states];
        let mut is_dead = vec![false; n_states];

        for (compact, &regex_id) in state_list.iter().enumerate() {
            is_match[compact] = dfa.is_match_state(regex_id);
            is_dead[compact] = dfa.is_dead_state(regex_id);
            for byte in 0..=255u8 {
                let next = dfa.next_state(regex_id, byte);
                byte_trans[compact * 256 + usize::from(byte)] = state_map[&next];
            }
            let next_eoi = dfa.next_eoi_state(regex_id);
            eoi_trans[compact] = state_map[&next_eoi];
        }

        let n_symbols = symbols.len();
        let mut symbol_trans = vec![0u32; n_states * n_symbols];
        for state in 0..n_states {
            for code in 0..n_symbols {
                let sym_bytes = symbols[code].to_u64().to_le_bytes();
                let sym_len = usize::from(symbol_lengths[code]);
                let mut s = u32::try_from(state).expect("MAX_STATES fits in u32");
                for &b in &sym_bytes[..sym_len] {
                    if is_match[s as usize] || is_dead[s as usize] {
                        break;
                    }
                    s = byte_trans[s as usize * 256 + usize::from(b)];
                }
                symbol_trans[state * n_symbols + code] = s;
            }
        }

        let start = state_map[&regex_start];

        Ok(Some(Self {
            n_symbols,
            symbol_trans,
            byte_trans,
            eoi_trans,
            is_match,
            is_dead,
            start,
        }))
    }

    /// Evaluate the regex against a single FSST code stream.
    ///
    /// Returns `true` if the decompressed bytes contain a match for the
    /// compiled regex (`is_match` semantics).
    pub(crate) fn matches(&self, codes: &[u8]) -> bool {
        let mut state = self.start;
        if self.is_match[state as usize] {
            return true;
        }
        if self.is_dead[state as usize] {
            // EOI on a dead start state is still dead.
            return false;
        }

        let mut i = 0;
        while i < codes.len() {
            let code = codes[i];
            i += 1;
            if code == ESCAPE_CODE {
                if i >= codes.len() {
                    // Malformed tail; treat the escape as no-op.
                    break;
                }
                let lit = codes[i];
                i += 1;
                state = self.byte_trans[state as usize * 256 + usize::from(lit)];
            } else {
                state = self.symbol_trans[state as usize * self.n_symbols + usize::from(code)];
            }
            if self.is_match[state as usize] {
                return true;
            }
            if self.is_dead[state as usize] {
                return false;
            }
        }

        // Some patterns (e.g. `^foo$`) only enter a match state after the
        // end-of-input transition fires.
        let final_state = self.eoi_trans[state as usize];
        self.is_match[final_state as usize]
    }
}
