// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! End-to-end coverage for regex predicate pushdown.
//!
//! These tests write a Vortex file containing a string column, then issue
//! DataFusion SQL queries with PostgreSQL-flavored regex operators (`~`,
//! `~*`, `!~`, `!~*`) to verify the predicate is converted into a
//! `vortex.regex` call and produces the expected rows.

use std::sync::Arc;

use datafusion::arrow::array::ArrayRef as ArrowArrayRef;
use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::array::StringArray;
use datafusion_common::assert_batches_sorted_eq;

use crate::common_tests::TestSessionContext;

async fn fixture() -> anyhow::Result<TestSessionContext> {
    let ctx = TestSessionContext::default();

    let names: ArrowArrayRef = Arc::new(StringArray::from(vec![
        Some("alpha"),
        Some("Alphabet"),
        Some("beta"),
        Some("gamma"),
        Some("alphanumeric"),
        None,
    ]));
    let batch = RecordBatch::try_from_iter(vec![("name", names)])?;
    ctx.write_arrow_batch("files/regex.vortex", &batch).await?;

    let schema = batch.schema();
    let provider = ctx
        .table_provider("regex_tbl", "/files/", schema.as_ref().clone())
        .await?;
    ctx.session.register_table("regex_tbl", provider)?;
    Ok(ctx)
}

#[tokio::test]
async fn regex_match_case_sensitive() -> anyhow::Result<()> {
    let ctx = fixture().await?;

    let batches = ctx
        .session
        .sql("SELECT name FROM regex_tbl WHERE name ~ '^alpha'")
        .await?
        .collect()
        .await?;

    assert_batches_sorted_eq!(
        &[
            "+--------------+",
            "| name         |",
            "+--------------+",
            "| alpha        |",
            "| alphanumeric |",
            "+--------------+",
        ],
        &batches
    );
    Ok(())
}

#[tokio::test]
async fn regex_match_case_insensitive() -> anyhow::Result<()> {
    let ctx = fixture().await?;

    let batches = ctx
        .session
        .sql("SELECT name FROM regex_tbl WHERE name ~* '^alpha'")
        .await?
        .collect()
        .await?;

    assert_batches_sorted_eq!(
        &[
            "+--------------+",
            "| name         |",
            "+--------------+",
            "| Alphabet     |",
            "| alpha        |",
            "| alphanumeric |",
            "+--------------+",
        ],
        &batches
    );
    Ok(())
}

#[tokio::test]
async fn regex_not_match_case_sensitive() -> anyhow::Result<()> {
    let ctx = fixture().await?;

    let batches = ctx
        .session
        .sql("SELECT name FROM regex_tbl WHERE name !~ '^alpha'")
        .await?
        .collect()
        .await?;

    assert_batches_sorted_eq!(
        &[
            "+----------+",
            "| name     |",
            "+----------+",
            "| Alphabet |",
            "| beta     |",
            "| gamma    |",
            "+----------+",
        ],
        &batches
    );
    Ok(())
}

#[tokio::test]
async fn regex_not_match_case_insensitive() -> anyhow::Result<()> {
    // The case-insensitive negated path is the trickiest one to get right —
    // it's easy to flip a sign on either flag.
    let ctx = fixture().await?;

    let batches = ctx
        .session
        .sql("SELECT name FROM regex_tbl WHERE name !~* '^alpha'")
        .await?
        .collect()
        .await?;

    assert_batches_sorted_eq!(
        &[
            "+-------+",
            "| name  |",
            "+-------+",
            "| beta  |",
            "| gamma |",
            "+-------+",
        ],
        &batches
    );
    Ok(())
}

#[tokio::test]
async fn regex_match_with_alternation() -> anyhow::Result<()> {
    let ctx = fixture().await?;

    let batches = ctx
        .session
        .sql("SELECT name FROM regex_tbl WHERE name ~ '^(beta|gamma)$'")
        .await?
        .collect()
        .await?;

    assert_batches_sorted_eq!(
        &[
            "+-------+",
            "| name  |",
            "+-------+",
            "| beta  |",
            "| gamma |",
            "+-------+",
        ],
        &batches
    );
    Ok(())
}
