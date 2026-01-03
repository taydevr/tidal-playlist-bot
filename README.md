# tidal-playlist-bot

Python CLI that integrates OpenAI and `tidalapi` to generate, curate, reorder, and maintain large-scale TIDAL playlists.  
Supports track-based playlists, full albums, and complete artist discographies, with advanced control over ordering, deduplication, pinned tracks, and playlist updates.

---

## Features

- 🎵 Create playlists using OpenAI
  - Curated track lists
  - Full albums
  - Complete artist discographies (studio albums)

- 🔁 Update existing playlists
  - Append new content
  - Merge & deduplicate
  - Replace playlist contents

- 🔀 Reorder / shuffle existing playlists (no OpenAI required)
  - Album ordering (artist/year, year/artist, as-is, shuffled)
  - Track mixing (keep album order, shuffle within albums, global shuffle)

- 📌 Pinned intro tracks
  - Fix specific tracks at the beginning of a playlist
  - Preserved during reordering and updates

- 🧹 Smart deduplication
  - Strict or relaxed modes
  - Avoid duplicate tracks across albums and discographies

- 🎚️ Version filtering
  - Include or exclude live versions
  - Include or exclude remasters
  - Include or exclude deluxe/expanded editions
  - Avoid compilations (Best Of / Greatest Hits)

- 🗑️ Delete playlists
  - Safely remove playlists from your TIDAL account with confirmation

- 💾 Designed for very large libraries
  - Thousands of tracks supported
  - Ideal for offline downloads on DAPs and large SD cards

---

## Requirements

- Python 3.10 or newer
- A TIDAL account
- An OpenAI API key

---

## Installation

Clone the repository:

```bash
git clone https://github.com/taydevr/tidal-playlist-bot.git
cd tidal-playlist-bot
````

Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate    # macOS / Linux

.venv\Scripts\activate       # Windows
````

Install dependencies:
```bash
pip install openai tidalapi
````

---

## Usage
Run the tool:
```bash
python tidal_playlist_bot.py
````

On first use, the script will:

- Ask for your **OpenAI API key** (input is hidden, not stored)
- Start a **TIDAL device login flow**
- Save a local **TIDAL token** (`tidal_token.json`) for future runs

---

## Main Actions

When launched, you can choose one of the following actions:

1. **Create / Update playlist using OpenAI**
2. **Reorder / Shuffle an existing playlist** (no OpenAI required)
3. **Delete a playlist**

---

## Content Modes

### Tracks Mode

- OpenAI generates a curated list of tracks
- Ordering options:
  - Curated (model order)
  - Shuffle
  - Sort by artist
  - Sort by year (if available)

---

### Albums Mode

- OpenAI selects full albums
- All tracks from each album are added
- Albums remain grouped unless you choose to shuffle tracks

---

### Discography Mode

- OpenAI selects artists
- The script fetches **studio albums per artist**
- All tracks from those albums are added
- Designed for building **complete artist libraries**

---

## Example Prompts

Below are example prompts you can copy and paste when using
Create / Update playlist using OpenAI.

---

## Reordering & Shuffling Existing Playlists

You can reorganize any existing playlist **without using OpenAI**.
```text
Create a curated playlist focused on 1980s rock hits.
Include the most iconic and widely recognized songs from the decade.
Prioritize studio versions and original releases.
Avoid live recordings, remasters when originals exist, and compilations if studio albums are available.
Balance mainstream classics with influential deep cuts that defined the sound of the 80s.
Keep the playlist cohesive and suitable for long offline listening sessions.
````

### Album Block Ordering

- Artist → Year → Album
- Year → Artist → Album
- As-is (keep current order)
- Shuffle albums (album blocks)

### Track Mixing

- Keep track order within albums
- Shuffle tracks within albums
- Global shuffle (all tracks fully randomized)

> ⚠️ Reordering **rebuilds the playlist** (clear + re-add).  
> A confirmation prompt is always shown before applying changes.

---

## Pinned Tracks

You can pin specific tracks so they always appear at the beginning of the playlist.

Example input:

```text
Artist - Track Title
Another Artist - Another Track
````

Pinned tracks:

- Always stay at the top
- Are deduplicated automatically
- Are preserved during reorders and updates

---

## TIDAL Authentication

- Uses TIDAL’s official **device authorization flow**
- On first login, a browser link is shown
- A local token file (`tidal_token.json`) is created
- Future runs reuse the token automatically

---

## Safety Notes

- No local music files are modified
- Playlist deletion and rebuild actions require explicit confirmation
- OpenAI is only used when generating new content
- Playlist pagination is handled to avoid track loss

---

## Disclaimer

This project is **not affiliated with or endorsed by TIDAL or OpenAI**.  
Use at your own risk and in compliance with the terms of service of all platforms involved.
