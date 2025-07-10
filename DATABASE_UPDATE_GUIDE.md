# 🎨 ArtiFactor Database Update Guide

This guide covers how to update the ArtiFactor database with new artist and artwork data from WikiArt.

## 📋 Table of Contents

- [Local Testing](#-local-testing)
- [Script Parameters](#-script-parameters)
- [Updating Live Server](#-updating-live-server)
- [Troubleshooting](#-troubleshooting)

---

## 🧪 Local Testing

### 1. Set Admin Password

First, set the admin password environment variable:

```bash
export FINAL_SQL_ADMIN_PASSWORD="Girimehkala"
```

### 2. Build/Run Knowledge Server

If you've updated the KnowledgeServer code:

```bash
./docker_build.sh
```

Otherwise, just run the existing container:

```bash
./docker_run.sh
```

### 3. Run Scraping Script

Execute the scraping script to gather new data:

```bash
python3 scrape_to_staging.py --limit 5000 --depth 1000 --clear false
```

### 4. Review Staged Data

When the script finishes, new JSON files will be created in `LOCALDB/staging/`.

Visit the staging review page:
```
http://localhost:8080/staging_review
```

Use this page to:
- Review each artist file
- Make edits as needed
- Validate data quality

### 5. Final Review & Commit

Go to the final review page:
```
http://localhost:8080/staging_review/final_sql_review
```

Click the buttons to commit the staged artist files to the database.

### 6. Update Embeddings

Run the embeddings update script:

```bash
python3 update_embeddings.py
```

🎉 **Congratulations!** Your local database has been updated!

---

## 🔧 Script Parameters

### `scrape_to_staging.py` Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `--limit` | Integer | Maximum number of artists to process | `--limit 5000` |
| `--depth` | Integer | Maximum number of artworks per artist | `--depth 1000` |
| `--clear` | Boolean | Whether to clear existing staging data | `--clear false` |
| `--download` | Boolean | Whether to download and process images | `--download true` |
| `--resume` | Boolean | Resume from last processed artist | `--resume true` |
| `--log-level` | String | Logging verbosity (DEBUG, INFO, WARNING, ERROR) | `--log-level INFO` |

### Usage Examples

```bash
# Basic scraping without clearing existing data
python3 scrape_to_staging.py --limit 1000 --depth 500 --clear false

# Full scraping with image download (for server)
python3 scrape_to_staging.py --limit 5000 --depth 1000 --clear true --download true

# Resume interrupted scraping
python3 scrape_to_staging.py --limit 5000 --depth 1000 --clear false --resume true

# Quick test run
python3 scrape_to_staging.py --limit 10 --depth 5 --clear false --download false
```

---

## 🌐 Updating Live Server

### 1. Update Data Server

First, follow the steps to update the data server:
[HOW TO UPDATE THE DATA SERVER](https://www.notion.so/HOW-TO-UPDATE-THE-DATA-SERVER-19d54b0f0b1f80379552f031ac2cc437?pvs=21)

### 2. Upload Updated Scripts (if needed)

If you've modified the scraping scripts locally:

```bash
gcloud compute scp scrape_to_staging.py resnet50wikiart:~/LOCALDB/
gcloud compute scp update_embeddings.py resnet50wikiart:~/LOCALDB/
```
or the artist names list
```bash
gcloud compute scp artist_names.txt resnet50wikiart:~/LOCALDB/
```

### 3. SSH into Server

```bash
gcloud compute ssh resnet50wikiart
```

### 4. Set Admin Password

```bash
export FINAL_SQL_ADMIN_PASSWORD="Girimehkala"
```

### 5. Run Test Scrape

Perform a small test to ensure everything works:

```bash
python3 scrape_to_staging.py --limit 30 --depth 10 --clear false --download true
```

### 6. Execute Full Scraping

Run the comprehensive scraping process:

```bash
python3 scrape_to_staging.py --limit 5000 --depth 1000 --clear true --download true && \
python3 scrape_to_staging.py --limit 5000 --depth 1000 --clear false --download true && \
python3 scrape_to_staging.py --limit 5000 --depth 1000 --clear false --download true
```

### 7. Monitor Progress

Wait several hours for completion. If the script breaks, rerun it with `--resume true`.

### 8. Review & Commit

1. Review staged data at `/staging_review`
2. Commit changes at `/staging_review/final_sql_review`
3. Run `update_embeddings.py`

---

## 🔍 Troubleshooting

### Common Issues

**Script Interrupted:**
```bash
# Resume from where it left off
python3 scrape_to_staging.py --limit 5000 --depth 1000 --clear false --resume true
```

**Memory Issues:**
```bash
# Reduce batch size
python3 scrape_to_staging.py --limit 1000 --depth 500 --clear false
```

**Network Timeouts:**
```bash
# Run with retry and smaller batches
python3 scrape_to_staging.py --limit 500 --depth 100 --clear false --download true
```

### Log Files

Check log files in `LOCALDB/` for detailed error information:
- `artist_summary_log.json` - Progress summary
- `log-[timestamp].txt` - Detailed execution logs

### Data Validation

Use the data cleaner tool to fix common issues:
```
http://localhost:8080/data_cleaner
```

Available cleaning options:
- Malformed JSON repair
- Artists without images
- Orphaned image references
- Artist ↔ Images integrity check
- Related keywords integrity

---

## 📝 Notes

- Always test locally before updating the live server
- The scraping process is resumable - interruptions won't lose progress
- Review staged data carefully before committing to the database
- Keep backups of the database before major updates
- Monitor server resources during large scraping operations

---

*Last updated: July 9, 2025*
