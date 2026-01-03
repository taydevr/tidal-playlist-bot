import os
import re
import json
import time
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import tidalapi
from openai import OpenAI
from datetime import datetime

DEFAULT_MODEL = "gpt-4.1-mini"
MAX_TRACKS_DEFAULT = 250
MAX_ALBUMS_DEFAULT = 40
MAX_ARTISTS_DEFAULT = 20


# -----------------------------
# Data models
# -----------------------------
@dataclass
class TrackCandidate:
    artist: str
    title: str
    album: Optional[str] = None
    year: Optional[int] = None
    note: Optional[str] = None


@dataclass
class AlbumCandidate:
    artist: str
    album: str
    year: Optional[int] = None
    note: Optional[str] = None


@dataclass
class ArtistCandidate:
    artist: str
    note: Optional[str] = None


@dataclass
class UserPrefs:
    # Content mode: "tracks" | "albums" | "discography"
    content_mode: str

    # Tracks-only
    max_tracks: int
    ordering: str  # "curated" | "shuffle" | "artist" | "year"

    # Albums/discography-only limits
    max_albums: int
    max_artists: int
    max_albums_per_artist: int

    # album ordering + track mixing for albums/discography
    album_ordering: str  # "artist_year" | "year_artist" | "as_is" | "shuffle_albums"
    track_mixing: str    # "album_order" | "shuffle_within_album" | "global_shuffle"

    # Version / filtering preferences
    include_live: bool
    include_deluxe: bool
    include_remaster: bool
    prefer_original: bool
    allow_compilations: bool

    # Dedupe
    avoid_duplicates: bool
    strict_title_artist_dedupe: bool

    # Matching UI
    show_preview_count: int
    low_conf_threshold: int

    # Playlist target / update behavior
    playlist_action: str   # "create" | "update"
    update_mode: str       # "append" | "merge_dedupe" | "replace"
    target_playlist_id: Optional[str] = None

    # NEW: Pinned intro (Create/Update)
    pin_intro: bool = False
    pinned_lines: List[str] = field(default_factory=list)


# -----------------------------
# Helpers
# -----------------------------
def normalize(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("’", "'")
    s = re.sub(r"[’'`]", "", s)
    s = re.sub(r"[^a-z0-9áéíóúñü\s\-&]", "", s)
    return s


def yn(prompt: str, default: bool = False) -> bool:
    suffix = " [Y/n] " if default else " [y/N] "
    ans = input(prompt + suffix).strip().lower()
    if not ans:
        return default
    return ans in ("y", "yes", "s", "si", "sí")


def pick(prompt: str, options: Dict[str, str], default_key: str) -> str:
    if prompt:
        print(prompt)
    for k, desc in options.items():
        print(f"  {k}) {desc}")
    ans = input(f"Choose ({'/'.join(options.keys())}) [default {default_key}]: ").strip().lower()
    return ans if ans in options else default_key


def get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Paste your OPENAI_API_KEY (not stored): ").strip()
    return OpenAI(api_key=api_key)


def choose_name_and_description(model_name: str, model_desc: str) -> Tuple[str, str]:
    """
    Lets user decide the playlist name/description:
      m) use model output
      e) enter their own
      s) pick one of 3 local suggestions, keep model description (editable afterward)
    """
    print("\nPlaylist naming:")
    mode = pick(
        "How do you want to set the playlist name/description?",
        {
            "m": "Use model-provided name/description",
            "e": "Enter my own name/description",
            "s": "Pick from 3 short name suggestions (keep model description)",
        },
        default_key="m",
    )

    model_name = (model_name or "").strip() or "Auto Playlist"
    model_desc = (model_desc or "").strip() or "Auto-generated from your prompt."

    if mode == "m":
        return model_name, model_desc

    if mode == "e":
        name = input("Playlist name: ").strip() or model_name
        desc = input("Playlist description: ").strip() or model_desc
        return name, desc

    # mode == "s"
    base = model_name
    suggestions = [
        base,
        f"{base} Library",
        f"{base} (Full Albums)",
    ]
    print("\nPick one:")
    for i, s in enumerate(suggestions, 1):
        print(f"  {i}) {s}")
    raw = input("Choose 1-3 (default 1): ").strip()
    try:
        idx = int(raw) if raw else 1
        idx = 1 if idx < 1 or idx > 3 else idx
    except ValueError:
        idx = 1

    chosen = suggestions[idx - 1]
    if yn("Edit description too?", default=False):
        desc = input("Playlist description: ").strip() or model_desc
        return chosen, desc
    return chosen, model_desc


# -----------------------------
# Prompt builders (GENRE-AGNOSTIC)
# -----------------------------
def build_system_prompt_tracks(prefs: UserPrefs) -> str:
    rules = [
        "You are a music curator.",
        "Follow the user's constraints exactly.",
        "Return ONLY valid JSON (no markdown) using this schema:",
        '{ "playlist_name": "string", "description": "string", "tracks": [ { "artist":"", "title":"", "album":null, "year":null, "note":null } ] }',
        f"Maximum {prefs.max_tracks} tracks.",
        "Avoid duplicates (same song).",
        "Prefer official releases over karaoke/tribute/cover versions.",
    ]
    if not prefs.include_live:
        rules.append("EXCLUDE live versions.")
    if not prefs.include_deluxe:
        rules.append("EXCLUDE deluxe/expanded editions.")
    if not prefs.include_remaster:
        rules.append("EXCLUDE remasters when an original version exists.")
    if prefs.prefer_original:
        rules.append("PREFER original studio album versions over remaster/deluxe/live.")
    if not prefs.allow_compilations:
        rules.append("Avoid compilations like 'Best Of' / 'Greatest Hits' if studio albums exist.")
    return "\n".join(rules)


def build_system_prompt_albums(prefs: UserPrefs) -> str:
    rules = [
        "You are a music curator.",
        "Follow the user's constraints exactly.",
        "Return ONLY valid JSON (no markdown) using this schema:",
        '{ "playlist_name": "string", "description": "string", "albums": [ { "artist":"", "album":"", "year":null, "note":null } ] }',
        f"Maximum {prefs.max_albums} albums total.",
        "Prioritize studio albums (not live).",
        "Avoid duplicates (same album).",
    ]
    if not prefs.include_live:
        rules.append("EXCLUDE live albums.")
    if not prefs.include_deluxe:
        rules.append("EXCLUDE deluxe/expanded editions.")
    if not prefs.include_remaster:
        rules.append("EXCLUDE remasters when an original album exists.")
    if prefs.prefer_original:
        rules.append("PREFER original studio albums over remaster/deluxe/live.")
    if not prefs.allow_compilations:
        rules.append("Avoid compilations like 'Best Of' / 'Greatest Hits'.")
    return "\n".join(rules)


def build_system_prompt_discography(prefs: UserPrefs) -> str:
    rules = [
        "You are a music curator.",
        "Follow the user's constraints exactly.",
        "Return ONLY valid JSON (no markdown) using this schema:",
        '{ "playlist_name": "string", "description": "string", "artists": [ { "artist":"", "note":null } ] }',
        f"Maximum {prefs.max_artists} artists.",
        "Choose artists with substantial studio discographies available on streaming platforms.",
        "Avoid duplicates (same artist).",
    ]
    return "\n".join(rules)


# -----------------------------
# OpenAI generators
# -----------------------------
def _safe_parse_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise RuntimeError("Model did not return parseable JSON.")
        return json.loads(m.group(0))


def generate_tracklist_with_openai(client: OpenAI, user_request: str, prefs: UserPrefs):
    system = build_system_prompt_tracks(prefs)
    resp = client.responses.create(
        model=DEFAULT_MODEL,
        input=[{"role": "system", "content": system}, {"role": "user", "content": user_request}],
    )
    data = _safe_parse_json(resp.output_text)

    playlist_name = data.get("playlist_name", "Auto Playlist")
    description = data.get("description", "Auto-generated from your prompt.")
    tracks: List[TrackCandidate] = []
    for item in data.get("tracks", []):
        if not item.get("artist") or not item.get("title"):
            continue
        tracks.append(
            TrackCandidate(
                artist=item["artist"],
                title=item["title"],
                album=item.get("album"),
                year=item.get("year"),
                note=item.get("note"),
            )
        )
    return playlist_name, description, tracks


def generate_albumlist_with_openai(client: OpenAI, user_request: str, prefs: UserPrefs):
    system = build_system_prompt_albums(prefs)
    resp = client.responses.create(
        model=DEFAULT_MODEL,
        input=[{"role": "system", "content": system}, {"role": "user", "content": user_request}],
    )
    data = _safe_parse_json(resp.output_text)

    playlist_name = data.get("playlist_name", "Auto Playlist")
    description = data.get("description", "Auto-generated from your prompt.")
    albums: List[AlbumCandidate] = []
    for item in data.get("albums", []):
        if not item.get("artist") or not item.get("album"):
            continue
        albums.append(
            AlbumCandidate(
                artist=item["artist"],
                album=item["album"],
                year=item.get("year"),
                note=item.get("note"),
            )
        )
    return playlist_name, description, albums


def generate_artistlist_with_openai(client: OpenAI, user_request: str, prefs: UserPrefs):
    system = build_system_prompt_discography(prefs)
    resp = client.responses.create(
        model=DEFAULT_MODEL,
        input=[{"role": "system", "content": system}, {"role": "user", "content": user_request}],
    )
    data = _safe_parse_json(resp.output_text)

    playlist_name = data.get("playlist_name", "Auto Playlist")
    description = data.get("description", "Auto-generated from your prompt.")
    artists: List[ArtistCandidate] = []
    for item in data.get("artists", []):
        if not item.get("artist"):
            continue
        artists.append(ArtistCandidate(artist=item["artist"], note=item.get("note")))
    return playlist_name, description, artists


def get_artist_albums(aobj) -> List[Any]:
    """Robust album fetch for tidalapi artist objects."""
    albums = []

    try:
        albums = list(aobj.get_albums())
        if albums:
            return albums
    except Exception:
        pass

    try:
        albums = list(aobj.albums(limit=200))
        if albums:
            return albums
    except Exception:
        pass

    try:
        albums = list(aobj.albums())
        if albums:
            return albums
    except Exception:
        pass

    try:
        albums = list(aobj.albums)
        if albums:
            return albums
    except Exception:
        pass

    return []


# -----------------------------
# TIDAL matching utilities
# -----------------------------
def is_live_title_or_album(name: str) -> bool:
    n = normalize(name)
    return any(x in n for x in [" live", "live at", "concert", "en vivo", "ao vivo"])


def is_deluxe(name: str) -> bool:
    n = normalize(name)
    return any(x in n for x in ["deluxe", "expanded", "anniversary", "special edition", "extended"])


def is_remaster(name: str) -> bool:
    n = normalize(name)
    return any(x in n for x in ["remaster", "remastered", "digitally remastered"])


def is_compilation(album_name: str) -> bool:
    n = normalize(album_name)
    return any(x in n for x in ["best of", "greatest hits", "collection", "anthology", "singles", "compilation"])


def score_track_match(candidate: TrackCandidate, tidal_track, prefs: UserPrefs) -> int:
    score = 0
    cand_artist = normalize(candidate.artist)
    cand_title = normalize(candidate.title)
    cand_album = normalize(candidate.album or "")

    tidal_title = normalize(getattr(tidal_track, "name", ""))
    tidal_album = normalize(getattr(getattr(tidal_track, "album", None), "name", "") or "")
    try:
        tidal_artist = normalize(tidal_track.artist.name)
    except Exception:
        tidal_artist = normalize(getattr(tidal_track, "artist", ""))

    if cand_title == tidal_title:
        score += 80
    elif cand_title and (cand_title in tidal_title or tidal_title in cand_title):
        score += 45

    if cand_artist == tidal_artist:
        score += 70
    elif cand_artist and (cand_artist in tidal_artist or tidal_artist in cand_artist):
        score += 35

    if cand_album and tidal_album:
        if cand_album == tidal_album:
            score += 25
        elif cand_album in tidal_album or tidal_album in cand_album:
            score += 10

    bad_words = ["karaoke", "tribute", "cover", "made famous", "instrumental"]
    if any(w in tidal_title for w in bad_words) or any(w in tidal_album for w in bad_words):
        score -= 80

    title_or_album = f"{tidal_title} {tidal_album}"

    if not prefs.include_live and is_live_title_or_album(title_or_album):
        score -= 999
    if not prefs.include_deluxe and is_deluxe(title_or_album):
        score -= 200
    if not prefs.include_remaster and is_remaster(title_or_album):
        score -= 200
    if not prefs.allow_compilations and is_compilation(tidal_album):
        score -= 150

    if prefs.prefer_original:
        if is_live_title_or_album(title_or_album):
            score -= 80
        if is_deluxe(title_or_album):
            score -= 40
        if is_remaster(title_or_album):
            score -= 25

    return score


def choose_best_tidal_track(session: tidalapi.Session, candidate: TrackCandidate, prefs: UserPrefs):
    # If artist blank, still search by title
    q = f"{candidate.artist} {candidate.title}".strip()
    results = session.search(q)
    tracks = results.get("tracks", [])
    if not tracks:
        return None, 0

    best = None
    best_score = -10_000
    for t in tracks[:15]:
        sc = score_track_match(candidate, t, prefs)
        if sc > best_score:
            best_score = sc
            best = t

    if best_score < -500:
        return None, best_score

    return best, best_score


def score_album_match(candidate: AlbumCandidate, tidal_album, prefs: UserPrefs) -> int:
    score = 0
    cand_artist = normalize(candidate.artist)
    cand_album = normalize(candidate.album)

    tidal_album_name = normalize(getattr(tidal_album, "name", ""))
    try:
        tidal_artist = normalize(tidal_album.artist.name)
    except Exception:
        tidal_artist = ""

    if cand_album == tidal_album_name:
        score += 85
    elif cand_album and (cand_album in tidal_album_name or tidal_album_name in cand_album):
        score += 45

    if cand_artist == tidal_artist:
        score += 70
    elif cand_artist and (cand_artist in tidal_artist or tidal_artist in cand_artist):
        score += 35

    title_or_album = tidal_album_name
    if not prefs.include_live and is_live_title_or_album(title_or_album):
        score -= 999
    if not prefs.include_deluxe and is_deluxe(title_or_album):
        score -= 200
    if not prefs.include_remaster and is_remaster(title_or_album):
        score -= 200
    if not prefs.allow_compilations and is_compilation(title_or_album):
        score -= 150

    if prefs.prefer_original:
        if is_live_title_or_album(title_or_album):
            score -= 80
        if is_deluxe(title_or_album):
            score -= 40
        if is_remaster(title_or_album):
            score -= 25

    return score


def choose_best_tidal_album(session: tidalapi.Session, candidate: AlbumCandidate, prefs: UserPrefs):
    q = f"{candidate.artist} {candidate.album}"
    results = session.search(q)
    albums = results.get("albums", [])
    if not albums:
        return None, 0

    best = None
    best_score = -10_000
    for a in albums[:15]:
        sc = score_album_match(candidate, a, prefs)
        if sc > best_score:
            best_score = sc
            best = a

    if best_score < -500:
        return None, best_score

    return best, best_score


def score_artist_match(candidate: ArtistCandidate, tidal_artist) -> int:
    cand = normalize(candidate.artist)
    try:
        name = normalize(tidal_artist.name)
    except Exception:
        name = ""

    if cand == name:
        return 100
    if cand and (cand in name or name in cand):
        return 60
    return 0


def choose_best_tidal_artist(session: tidalapi.Session, candidate: ArtistCandidate):
    q = candidate.artist
    results = session.search(q)
    artists = results.get("artists", [])
    if not artists:
        return None, 0

    best = None
    best_score = -1
    for a in artists[:10]:
        sc = score_artist_match(candidate, a)
        if sc > best_score:
            best_score = sc
            best = a
    return best, best_score


def get_album_tracks(album_obj) -> List[Any]:
    try:
        t = album_obj.tracks()
        return list(t)
    except Exception:
        try:
            return list(album_obj.tracks)
        except Exception:
            return []


def filter_album_obj(album_obj, prefs: UserPrefs) -> bool:
    name = normalize(getattr(album_obj, "name", ""))
    if not prefs.include_live and is_live_title_or_album(name):
        return False
    if not prefs.include_deluxe and is_deluxe(name):
        return False
    if not prefs.include_remaster and is_remaster(name):
        return False
    if not prefs.allow_compilations and is_compilation(name):
        return False
    return True


# -----------------------------
# NEW: Pinned intro helpers (scoring-based)
# -----------------------------
def collect_pinned_lines_ui() -> List[str]:
    pinned: List[str] = []
    if not yn("\nPin specific tracks at the BEGINNING of the playlist?", default=False):
        return pinned
    print("Paste pinned tracks (one per line). Examples:")
    print("  Artist - Track Title")
    print("  Another Artist - Another Title")
    print("Empty line to finish.\n")
    while True:
        line = input("> ").strip()
        if not line:
            break
        pinned.append(line)
    return pinned


def parse_pinned_line(line: str) -> TrackCandidate:
    # Expected: "Artist - Title"
    if " - " in line:
        a, t = line.split(" - ", 1)
        return TrackCandidate(artist=a.strip(), title=t.strip())
    # Fallback: search by whole line
    return TrackCandidate(artist="", title=line.strip())


def resolve_pinned_intro_ids(session: tidalapi.Session, pinned_lines: List[str], prefs: UserPrefs) -> Tuple[List[int], List[str]]:
    """
    Resolve pinned tracks using choose_best_tidal_track scoring (top-15).
    Respects prefs filtering since score_track_match penalizes forbidden versions.
    Returns (resolved_ids_in_order, not_found_lines).
    """
    resolved_ids: List[int] = []
    not_found: List[str] = []

    # For pinned, be slightly tolerant but still quality-controlled
    min_score = max(60, prefs.low_conf_threshold - 10)

    for line in pinned_lines:
        cand = parse_pinned_line(line)
        if not cand.artist:
            # title-only fallback search (still try top scoring by wrapping as candidate)
            t, sc = choose_best_tidal_track(session, cand, prefs)
        else:
            t, sc = choose_best_tidal_track(session, cand, prefs)

        if t is None or sc < min_score:
            not_found.append(line)
            continue
        resolved_ids.append(t.id)

    # dedupe while preserving order
    seen = set()
    out: List[int] = []
    for tid in resolved_ids:
        if tid in seen:
            continue
        seen.add(tid)
        out.append(tid)

    return out, not_found


def apply_pins_to_ids(pinned_ids: List[int], ids: List[int]) -> List[int]:
    pinned_set = set(pinned_ids)
    rest = [tid for tid in ids if tid not in pinned_set]
    return list(pinned_ids) + rest


# -----------------------------
# Album ordering + track mixing (for albums/discography mode)
# -----------------------------
def album_artist_name(album_obj) -> str:
    try:
        return getattr(album_obj.artist, "name", "") or ""
    except Exception:
        return ""


def album_year(album_obj) -> Optional[int]:
    for attr in ("year", "release_year", "releaseYear", "releaseDate", "release_date"):
        try:
            v = getattr(album_obj, attr)
        except Exception:
            v = None
        if not v:
            continue
        if isinstance(v, int):
            return v
        if isinstance(v, datetime):
            return v.year
        if isinstance(v, str):
            m = re.search(r"(\d{4})", v)
            if m:
                return int(m.group(1))
    try:
        rd = getattr(album_obj, "release_date", None)
        if isinstance(rd, datetime):
            return rd.year
    except Exception:
        pass
    return None


def order_album_entries(album_entries: List[Tuple[Any, List[Any]]], prefs: UserPrefs) -> List[Tuple[Any, List[Any]]]:
    mode = (prefs.album_ordering or "").strip().lower()
    entries = album_entries[:]

    if mode == "shuffle_albums":
        random.shuffle(entries)
        return entries

    if mode == "as_is":
        return entries

    def k_artist_year(e):
        alb = e[0]
        y = album_year(alb)
        return (normalize(album_artist_name(alb)), y is None, y or 9999, normalize(getattr(alb, "name", "")))

    def k_year_artist(e):
        alb = e[0]
        y = album_year(alb)
        return (y is None, y or 9999, normalize(album_artist_name(alb)), normalize(getattr(alb, "name", "")))

    if mode == "artist_year":
        return sorted(entries, key=k_artist_year)
    if mode == "year_artist":
        return sorted(entries, key=k_year_artist)

    return entries


def flatten_album_entries_to_track_ids(album_entries: List[Tuple[Any, List[Any]]], prefs: UserPrefs) -> List[int]:
    ordered = order_album_entries(album_entries, prefs)
    mix = (prefs.track_mixing or "").strip().lower()

    if mix == "global_shuffle":
        all_tracks = []
        for _, tracks in ordered:
            all_tracks.extend(list(tracks))
        random.shuffle(all_tracks)
        return [t.id for t in all_tracks]

    out = []
    for _, tracks in ordered:
        tlist = list(tracks)
        if mix == "shuffle_within_album":
            random.shuffle(tlist)
        out.extend([t.id for t in tlist])
    return out


# -----------------------------
# Dedupe / ordering for tracks mode
# -----------------------------
def dedupe_candidates(cands: List[TrackCandidate], prefs: UserPrefs) -> List[TrackCandidate]:
    if not prefs.avoid_duplicates:
        return cands

    seen = set()
    out = []
    for c in cands:
        if prefs.strict_title_artist_dedupe:
            key = (normalize(c.artist), normalize(c.title))
        else:
            title = normalize(c.title)
            title = re.sub(r"\b(remaster(ed)?|live|deluxe|version|edit)\b", "", title).strip()
            key = (normalize(c.artist), title)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def order_candidates(cands: List[TrackCandidate], prefs: UserPrefs) -> List[TrackCandidate]:
    if prefs.ordering == "shuffle":
        c = cands[:]
        random.shuffle(c)
        return c
    if prefs.ordering == "artist":
        return sorted(cands, key=lambda x: (normalize(x.artist), normalize(x.title)))
    if prefs.ordering == "year":
        return sorted(cands, key=lambda x: (x.year is None, x.year or 9999, normalize(x.artist), normalize(x.title)))
    return cands  # curated


# -----------------------------
# TIDAL auth
# -----------------------------
def tidal_login_or_load(session: tidalapi.Session, token_path: str = "tidal_token.json") -> None:
    if os.path.exists(token_path):
        try:
            with open(token_path, "r", encoding="utf-8") as f:
                tok = json.load(f)

            expiry = tok.get("expiry_time")
            expiry_dt = datetime.fromisoformat(expiry) if expiry else None

            ok = session.load_oauth_session(
                tok["token_type"], tok["access_token"], tok.get("refresh_token"), expiry_dt
            )
            if ok:
                print("✅ TIDAL session loaded from token.")
                return
        except Exception:
            pass

    print("🔐 TIDAL login: a browser/device authorization flow will be shown in the console.")
    session.login_oauth_simple()
    if not session.check_login():
        raise RuntimeError("Could not complete TIDAL login.")

    tok = {
        "token_type": session.token_type,
        "access_token": session.access_token,
        "refresh_token": session.refresh_token,
        "expiry_time": session.expiry_time.isoformat() if session.expiry_time else None,
    }
    with open(token_path, "w", encoding="utf-8") as f:
        json.dump(tok, f, indent=2)
    print(f"✅ Token saved to {token_path} (keep it safe).")


# -----------------------------
# Playlist helpers (list/select/clear/update)
# -----------------------------
def list_user_playlists(session: tidalapi.Session, limit: int = 200):
    try:
        pls = list(session.user.playlists())
    except Exception:
        pls = []
    return pls[:limit]


def choose_playlist_interactively(session: tidalapi.Session):
    pls = list_user_playlists(session)
    if not pls:
        print("No playlists found on this account.")
        return None

    print("\nSelect a playlist:")
    for i, p in enumerate(pls, 1):
        name = getattr(p, "name", "")
        print(f"  {i:02d}) {name}")

    while True:
        raw = input(f"Enter number (1-{len(pls)}) or blank to cancel: ").strip()
        if not raw:
            return None
        try:
            idx = int(raw)
            if 1 <= idx <= len(pls):
                return pls[idx - 1]
        except ValueError:
            pass
        print("Invalid selection.")


def get_playlist_tracks(playlist, page_size: int = 100) -> List[Any]:
    """
    Fetch ALL tracks from a playlist (handles pagination in many tidalapi versions).
    """
    all_tracks: List[Any] = []

    # Try pagination: playlist.tracks(limit=?, offset=?)
    try:
        offset = 0
        while True:
            page = playlist.tracks(limit=page_size, offset=offset)
            page_list = list(page) if page else []
            if not page_list:
                break
            all_tracks.extend(page_list)
            if len(page_list) < page_size:
                break
            offset += page_size
        if all_tracks:
            return all_tracks
    except TypeError:
        pass
    except Exception:
        pass

    # Fallback: iterator
    try:
        return list(playlist.tracks())
    except Exception:
        pass

    # Last fallback: property
    try:
        return list(playlist.tracks)
    except Exception:
        return []


def get_playlist_track_ids(playlist) -> List[int]:
    return [t.id for t in get_playlist_tracks(playlist)]


def clear_playlist(playlist) -> None:
    try:
        playlist.clear()
        return
    except Exception:
        pass

    ids = get_playlist_track_ids(playlist)
    if not ids:
        return

    BATCH = 50
    for i in range(0, len(ids), BATCH):
        batch = ids[i:i + BATCH]
        try:
            playlist.remove(batch)
        except Exception:
            raise RuntimeError("Could not clear playlist with this tidalapi version.")


def apply_update_mode(existing_ids: List[int], new_ids: List[int], update_mode: str) -> List[int]:
    existing_set = set(existing_ids)
    if update_mode == "append":
        return [tid for tid in new_ids if tid not in existing_set]
    if update_mode == "merge_dedupe":
        out = list(existing_ids)
        for tid in new_ids:
            if tid not in existing_set:
                out.append(tid)
                existing_set.add(tid)
        return out
    if update_mode == "replace":
        return list(new_ids)
    return list(new_ids)


# -----------------------------
# Delete playlist flow
# -----------------------------
def delete_playlist_flow(session: tidalapi.Session):
    pl = choose_playlist_interactively(session)
    if not pl:
        print("Cancelled.")
        return

    name = getattr(pl, "name", "")
    print(f"\n⚠️ You are about to DELETE playlist: {name}")
    confirm = input("Type DELETE to confirm: ").strip()
    if confirm != "DELETE":
        print("Cancelled.")
        return

    try:
        pl.delete()
        print("✅ Playlist deleted.")
    except Exception as e:
        print(f"❌ Could not delete playlist (tidalapi limitation?). Error: {e}")


# -----------------------------
# Reorder / Shuffle existing playlist + pinned intro tracks (scoring-based)
# -----------------------------
def collect_pinned_queries() -> List[str]:
    # Backward-compatible name, but uses same UI as create/update
    return collect_pinned_lines_ui()


def resolve_pinned_ids(session: tidalapi.Session, pinned_queries: List[str], prefs: UserPrefs) -> Tuple[List[int], List[str]]:
    # Backward-compatible name
    return resolve_pinned_intro_ids(session, pinned_queries, prefs)


def track_artist_name(t) -> str:
    try:
        return t.artist.name or ""
    except Exception:
        try:
            return getattr(t, "artist", "") or ""
        except Exception:
            return ""


def track_album_name(t) -> str:
    try:
        alb = getattr(t, "album", None)
        return getattr(alb, "name", "") or ""
    except Exception:
        return ""


def track_year(t) -> Optional[int]:
    alb = None
    try:
        alb = getattr(t, "album", None)
    except Exception:
        alb = None
    if not alb:
        return None
    for attr in ("year", "release_year", "releaseYear", "releaseDate", "release_date"):
        try:
            v = getattr(alb, attr)
        except Exception:
            v = None
        if not v:
            continue
        if isinstance(v, int):
            return v
        if isinstance(v, datetime):
            return v.year
        if isinstance(v, str):
            m = re.search(r"(\d{4})", v)
            if m:
                return int(m.group(1))
    try:
        rd = getattr(alb, "release_date", None)
        if isinstance(rd, datetime):
            return rd.year
    except Exception:
        pass
    return None


def reorder_existing_playlist_flow(session: tidalapi.Session):
    pl = choose_playlist_interactively(session)
    if not pl:
        print("Cancelled.")
        return

    name = getattr(pl, "name", "")
    tracks = get_playlist_tracks(pl)
    print(f"DEBUG fetched tracks from playlist: {len(tracks)}")
    if not tracks:
        print("Playlist is empty. Nothing to reorder.")
        return

    # Use a prefs-like object to apply your version filters when resolving pins.
    # We default to permissive (so pins resolve) but still avoid covers/tribute via score_track_match.
    reorder_prefs = UserPrefs(
        content_mode="tracks",
        max_tracks=0,
        ordering="curated",
        max_albums=0,
        max_artists=0,
        max_albums_per_artist=0,
        album_ordering="as_is",
        track_mixing="album_order",
        include_live=True,
        include_deluxe=True,
        include_remaster=True,
        prefer_original=False,
        allow_compilations=True,
        avoid_duplicates=False,
        strict_title_artist_dedupe=True,
        show_preview_count=0,
        low_conf_threshold=70,
        playlist_action="update",
        update_mode="append",
        target_playlist_id=None,
        pin_intro=False,
        pinned_lines=[],
    )

    pinned_queries = collect_pinned_queries()
    pinned_ids, pinned_nf = resolve_pinned_ids(session, pinned_queries, reorder_prefs)

    if pinned_nf:
        print("\n⚠️ Some pinned queries could not be resolved (they will be skipped):")
        for q in pinned_nf[:20]:
            print(f"  - {q}")

    print("\nReorder mode (two steps):")
    order_key = pick(
        "1) How should ALBUM BLOCKS be ordered?",
        {
            "a": "Artist → Year → Album",
            "b": "Year → Artist → Album",
            "c": "As-is (keep current playlist order as the base)",
            "d": "Shuffle albums (album blocks random)",
        },
        default_key="a",
    )
    album_order_map = {"a": "artist_year", "b": "year_artist", "c": "as_is", "d": "shuffle_albums"}
    album_ordering = album_order_map[order_key]

    mix_key = pick(
        "2) How should TRACKS be mixed?",
        {
            "1": "Keep track order within each album (recommended)",
            "2": "Shuffle tracks within each album",
            "3": "GLOBAL shuffle (all tracks across all albums fully random)",
        },
        default_key="1",
    )
    track_mix_map = {"1": "album_order", "2": "shuffle_within_album", "3": "global_shuffle"}
    track_mixing = track_mix_map[mix_key]

    # Build album blocks from existing playlist tracks
    album_blocks: Dict[Tuple[str, int, str], List[Any]] = {}
    album_first_seen: Dict[Tuple[str, int, str], int] = {}

    for idx, t in enumerate(tracks):
        a = track_artist_name(t)
        y = track_year(t) or 9999
        al = track_album_name(t) or "Unknown Album"
        k = (a, y, al)
        album_blocks.setdefault(k, []).append(t)
        if k not in album_first_seen:
            album_first_seen[k] = idx

    entries = [(k, v) for k, v in album_blocks.items()]

    if album_ordering == "shuffle_albums":
        random.shuffle(entries)
    elif album_ordering == "as_is":
        entries.sort(key=lambda kv: album_first_seen.get(kv[0], 10**9))
    elif album_ordering == "artist_year":
        entries.sort(key=lambda kv: (normalize(kv[0][0]), kv[0][1] == 9999, kv[0][1], normalize(kv[0][2])))
    elif album_ordering == "year_artist":
        entries.sort(key=lambda kv: (kv[0][1] == 9999, kv[0][1], normalize(kv[0][0]), normalize(kv[0][2])))

    if track_mixing == "global_shuffle":
        all_tracks = []
        for _, ts in entries:
            all_tracks.extend(ts)
        random.shuffle(all_tracks)
        reordered_ids = [t.id for t in all_tracks]
    else:
        reordered_ids = []
        for _, ts in entries:
            tlist = list(ts)
            if track_mixing == "shuffle_within_album":
                random.shuffle(tlist)
            reordered_ids.extend([t.id for t in tlist])

    final_ids = apply_pins_to_ids(pinned_ids, reordered_ids) if pinned_ids else reordered_ids

    print("\n=== Reorder preview ===")
    print(f"Playlist: {name}")
    print(f"Tracks: {len(tracks)} → {len(final_ids)} (pinned: {len(pinned_ids)})")
    print(f"Album ordering: {album_ordering} | Track mixing: {track_mixing}")

    if pinned_ids:
        print("\nPinned intro (first up to 10):")
        for i, tid in enumerate(pinned_ids[:10], 1):
            print(f"  {i:02d}. track_id={tid}")

    if not yn("\n⚠️ This will REBUILD the playlist order (clear + re-add). Proceed?", default=False):
        print("Cancelled.")
        return

    clear_playlist(pl)

    BATCH = 50
    for i in range(0, len(final_ids), BATCH):
        pl.add(final_ids[i:i + BATCH])
        time.sleep(0.15)
        print(f"➕ Added {min(i + BATCH, len(final_ids))}/{len(final_ids)}")

    print("\n✅ Playlist reordered successfully.")


# -----------------------------
# Wizard (Create/Update only)
# -----------------------------
def collect_prefs() -> UserPrefs:
    content_mode = pick(
        "What do you want to add to the playlist?",
        {
            "t": "Tracks (curated list)",
            "a": "Albums (full albums curated by OpenAI)",
            "d": "Discography (studio albums per artist)",
        },
        default_key="t",
    )
    content_mode_map = {"t": "tracks", "a": "albums", "d": "discography"}

    max_tracks = MAX_TRACKS_DEFAULT
    ordering = "curated"

    max_albums = MAX_ALBUMS_DEFAULT
    max_artists = MAX_ARTISTS_DEFAULT
    max_albums_per_artist = 6

    album_ordering = "artist_year"
    track_mixing = "album_order"

    if content_mode_map[content_mode] == "tracks":
        max_tracks_str = input(f"Maximum tracks? (enter={MAX_TRACKS_DEFAULT}): ").strip()
        max_tracks = int(max_tracks_str) if max_tracks_str else MAX_TRACKS_DEFAULT

        ordering_key = pick(
            "How should tracks be ordered?",
            {
                "c": "Curated (model order)",
                "s": "Shuffle from the start",
                "a": "Sort by artist",
                "y": "Sort by year (if provided)",
            },
            default_key="c",
        )
        ordering_map = {"c": "curated", "s": "shuffle", "a": "artist", "y": "year"}
        ordering = ordering_map[ordering_key]
    else:
        max_albums_str = input(f"Maximum albums total? (enter={MAX_ALBUMS_DEFAULT}): ").strip()
        max_albums = int(max_albums_str) if max_albums_str else MAX_ALBUMS_DEFAULT

        if content_mode_map[content_mode] == "discography":
            max_artists_str = input(f"Maximum artists? (enter={MAX_ARTISTS_DEFAULT}): ").strip()
            max_artists = int(max_artists_str) if max_artists_str else MAX_ARTISTS_DEFAULT

            per_artist_str = input("Max studio albums per artist? (enter=6): ").strip()
            max_albums_per_artist = int(per_artist_str) if per_artist_str else 6

        order_key = pick(
            "How should ALBUMS be ordered?",
            {
                "a": "Artist → Year → Album",
                "b": "Year → Artist → Album",
                "c": "As fetched (no sorting)",
                "d": "Shuffle albums (album blocks)",
            },
            default_key="a",
        )
        album_order_map = {"a": "artist_year", "b": "year_artist", "c": "as_is", "d": "shuffle_albums"}
        album_ordering = album_order_map[order_key]

        mix_key = pick(
            "How should TRACKS be mixed?",
            {
                "1": "Keep track order within each album (recommended)",
                "2": "Shuffle tracks within each album (optional)",
                "3": "GLOBAL shuffle (all tracks across all albums fully random)",
            },
            default_key="1",
        )
        track_mix_map = {"1": "album_order", "2": "shuffle_within_album", "3": "global_shuffle"}
        track_mixing = track_mix_map[mix_key]

    print("\nVersion preferences:")
    include_live = yn("Include LIVE versions/albums?", default=False)
    include_deluxe = yn("Include DELUXE/EXPANDED editions?", default=False)
    include_remaster = yn("Include REMASTER?", default=True)
    prefer_original = yn("Prefer ORIGINAL studio versions over remaster/deluxe/live?", default=True)
    allow_compilations = yn("Allow compilations (Best Of/Greatest Hits)?", default=False)

    print("\nAvoid duplicates:")
    avoid_duplicates = yn("Remove duplicates?", default=True)
    strict = yn("Strict dedupe by ARTIST+TITLE (tracks mode)?", default=True)

    preview_str = input("How many items to show in preview? (enter=25): ").strip()
    preview_count = int(preview_str) if preview_str else 25

    low_conf_str = input("Low-confidence threshold (score) (enter=70): ").strip()
    low_conf_threshold = int(low_conf_str) if low_conf_str else 70

    action_key = pick(
        "\nWhat do you want to do with the playlist?",
        {"n": "Create a new playlist", "u": "Update an existing playlist"},
        default_key="n",
    )
    playlist_action = "create" if action_key == "n" else "update"

    update_mode = "append"
    if playlist_action == "update":
        mode_key = pick(
            "How should the existing playlist be updated?",
            {
                "a": "Append only (keep existing, add new)",
                "m": "Merge & deduplicate (rebuild playlist: existing first, then new uniques)",
                "r": "Replace completely (delete all tracks, then add new)",
            },
            default_key="a",
        )
        update_mode_map = {"a": "append", "m": "merge_dedupe", "r": "replace"}
        update_mode = update_mode_map[mode_key]

    # NEW: pinned intro (for create/update)
    pinned_lines = collect_pinned_lines_ui()
    pin_intro = bool(pinned_lines)

    return UserPrefs(
        content_mode=content_mode_map[content_mode],
        max_tracks=max_tracks,
        ordering=ordering,
        max_albums=max_albums,
        max_artists=max_artists,
        max_albums_per_artist=max_albums_per_artist,
        album_ordering=album_ordering,
        track_mixing=track_mixing,
        include_live=include_live,
        include_deluxe=include_deluxe,
        include_remaster=include_remaster,
        prefer_original=prefer_original,
        allow_compilations=allow_compilations,
        avoid_duplicates=avoid_duplicates,
        strict_title_artist_dedupe=strict,
        show_preview_count=preview_count,
        low_conf_threshold=low_conf_threshold,
        playlist_action=playlist_action,
        update_mode=update_mode,
        target_playlist_id=None,
        pin_intro=pin_intro,
        pinned_lines=pinned_lines,
    )


# -----------------------------
# Main
# -----------------------------
def main():
    print("=== TIDAL Playlist Bot (OpenAI + tidalapi) ===\n")

    session = tidalapi.Session()
    tidal_login_or_load(session)

    app_action = pick(
        "\nWhat do you want to do?",
        {
            "1": "Create / Update playlist using OpenAI (generate content)",
            "2": "Reorder / Shuffle an existing playlist (no OpenAI)",
            "3": "Delete a playlist",
        },
        default_key="1",
    )

    if app_action == "3":
        delete_playlist_flow(session)
        return

    if app_action == "2":
        reorder_existing_playlist_flow(session)
        return

    user_request = input("\nEnter your request/prompt:\n> ").strip()

    prefs = collect_prefs()
    print(f"\nDEBUG album_ordering = {prefs.album_ordering!r} | track_mixing = {prefs.track_mixing!r}")

    client = get_openai_client()

    playlist_name = "Auto Playlist"
    description = "Auto-generated from your prompt."
    final_track_ids: List[int] = []

    # MODE: TRACKS
    if prefs.content_mode == "tracks":
        print("\n🤖 Generating TRACK list with OpenAI...\n")
        playlist_name, description, candidates = generate_tracklist_with_openai(client, user_request, prefs)

        if not candidates:
            print("No tracks were generated. Try a more specific prompt.")
            return

        candidates = dedupe_candidates(candidates, prefs)
        candidates = order_candidates(candidates, prefs)

        not_found: List[TrackCandidate] = []
        low_conf: List[Tuple[TrackCandidate, int]] = []
        resolved: List[Tuple[TrackCandidate, object, int]] = []

        print("\n🔎 Validating against TIDAL catalog (resolving tracks)...")
        for idx, c in enumerate(candidates, 1):
            t, sc = choose_best_tidal_track(session, c, prefs)
            if t is None:
                not_found.append(c)
                continue
            if sc < prefs.low_conf_threshold:
                low_conf.append((c, sc))
            resolved.append((c, t, sc))

            if idx % 25 == 0:
                print(f"  ...{idx}/{len(candidates)}")

        print("\n=== TIDAL-validated preview (tracks) ===")
        n = min(prefs.show_preview_count, len(resolved))
        print(f"✅ Found: {len(resolved)} | ❌ Not found: {len(not_found)} | ⚠️ Low confidence: {len(low_conf)}\n")

        print(f"Showing first {n} resolved tracks:")
        for i, (c, t, sc) in enumerate(resolved[:n], 1):
            try:
                artist = t.artist.name
            except Exception:
                artist = c.artist
            title = getattr(t, "name", c.title)
            album = getattr(getattr(t, "album", None), "name", "") or ""
            print(f"{i:02d}. ✅ ({sc}) {artist} - {title}" + (f" — {album}" if album else ""))

        if not resolved:
            print("\nNo tracks could be resolved in TIDAL. Nothing to do.")
            return

        if not_found:
            if not yn("\nSome tracks were NOT found. Continue using only resolved tracks?", default=True):
                print("Cancelled.")
                return

        final_track_ids = [t.id for (_, t, _) in resolved]

    # MODE: ALBUMS
    elif prefs.content_mode == "albums":
        print("\n🤖 Generating ALBUM list with OpenAI...\n")
        playlist_name, description, album_candidates = generate_albumlist_with_openai(client, user_request, prefs)

        if not album_candidates:
            print("No albums were generated. Try a more specific prompt.")
            return

        not_found: List[AlbumCandidate] = []
        resolved_albums: List[Tuple[AlbumCandidate, object, int]] = []

        print("\n🔎 Validating against TIDAL catalog (resolving albums)...")
        for idx, a in enumerate(album_candidates, 1):
            alb, sc = choose_best_tidal_album(session, a, prefs)
            if alb is None:
                not_found.append(a)
                continue
            resolved_albums.append((a, alb, sc))
            if idx % 10 == 0:
                print(f"  ...{idx}/{len(album_candidates)}")

        album_entries: List[Tuple[Any, List[Any]]] = []
        for (_, alb_obj, _) in resolved_albums:
            if not filter_album_obj(alb_obj, prefs):
                continue
            tracks = get_album_tracks(alb_obj)
            if not tracks:
                continue
            album_entries.append((alb_obj, tracks))

        if not album_entries:
            print("\nNo albums could be expanded into tracks. Nothing to do.")
            return

        final_track_ids = flatten_album_entries_to_track_ids(album_entries, prefs)

    # MODE: DISCOGRAPHY
    else:
        print("\n🤖 Generating ARTIST list for discography with OpenAI...\n")
        playlist_name, description, artists = generate_artistlist_with_openai(client, user_request, prefs)

        if not artists:
            print("No artists were generated. Try a more specific prompt.")
            return

        resolved_artists: List[Any] = []
        print("\n🔎 Resolving artists in TIDAL...")
        for idx, ac in enumerate(artists, 1):
            aobj, sc = choose_best_tidal_artist(session, ac)
            if aobj is None or sc < 40:
                continue
            resolved_artists.append(aobj)
            if idx % 10 == 0:
                print(f"  ...{idx}/{len(artists)}")

        if not resolved_artists:
            print("\nNo artists could be resolved in TIDAL. Nothing to do.")
            return

        album_entries: List[Tuple[Any, List[Any]]] = []
        total_albums_added = 0

        print("\n📚 Fetching studio albums per artist (discography mode)...")
        for aobj in resolved_artists:
            albums = get_artist_albums(aobj)
            filtered = [al for al in albums if filter_album_obj(al, prefs)]
            filtered = filtered[: prefs.max_albums_per_artist]

            for al in filtered:
                if total_albums_added >= prefs.max_albums:
                    break
                tracks = get_album_tracks(al)
                if not tracks:
                    continue
                album_entries.append((al, tracks))
                total_albums_added += 1

            if total_albums_added >= prefs.max_albums:
                break

        if not album_entries:
            print("\nNo albums could be collected/expanded. Nothing to do.")
            return

        final_track_ids = flatten_album_entries_to_track_ids(album_entries, prefs)

    # -------------------------
    # Name/Description choice (NEW, before create/update)
    # -------------------------
    playlist_name, description = choose_name_and_description(playlist_name, description)

    if not final_track_ids:
        print("\nNo tracks to add. Exiting.")
        return

    # -------------------------
    # Resolve pinned intro (Create/Update flow)
    # -------------------------
    pinned_ids: List[int] = []
    if prefs.pin_intro and prefs.pinned_lines:
        pinned_ids, pinned_nf = resolve_pinned_intro_ids(session, prefs.pinned_lines, prefs)
        if pinned_nf:
            print("\n⚠️ Some pinned tracks could not be resolved (skipped):")
            for x in pinned_nf[:20]:
                print(f"  - {x}")

    # -------------------------
    # Create OR Update playlist
    # -------------------------
    if prefs.playlist_action == "update":
        target_playlist = choose_playlist_interactively(session)
        if target_playlist is None:
            print("Cancelled (no playlist selected).")
            return

        existing_ids = get_playlist_track_ids(target_playlist)
        print(f"\nExisting tracks in playlist: {len(existing_ids)}")
        print(f"Newly resolved tracks: {len(final_track_ids)}")

        # Build final_ids according to update_mode
        if prefs.update_mode == "replace":
            confirm = input("⚠️ This will DELETE all existing tracks from the playlist. Type DELETE to confirm: ").strip()
            if confirm != "DELETE":
                print("Cancelled.")
                return
            final_ids = list(final_track_ids)
            # If pins, enforce pinned-first ordering by rebuilding
            if pinned_ids:
                final_ids = apply_pins_to_ids(pinned_ids, final_ids)
                clear_playlist(target_playlist)

        else:
            if prefs.update_mode == "append":
                # append-only normally adds just new tracks; BUT if pinned_ids exist, we must rebuild to force order.
                new_only = apply_update_mode(existing_ids, final_track_ids, "append")
                if not new_only:
                    # Still might want to pin? If user pinned, we can rebuild even without new tracks.
                    if pinned_ids:
                        rebuilt = apply_pins_to_ids(pinned_ids, existing_ids)
                        print("\n📌 Pinned intro enabled → rebuilding playlist order (no new tracks, order only).")
                        clear_playlist(target_playlist)
                        final_ids = rebuilt
                    else:
                        print("\nNothing new to add (already up to date).")
                        return
                else:
                    if pinned_ids:
                        rebuilt = existing_ids + new_only
                        final_ids = apply_pins_to_ids(pinned_ids, rebuilt)
                        print("\n📌 Pinned intro enabled → rebuilding playlist order (clear + re-add).")
                        clear_playlist(target_playlist)
                    else:
                        # true append: just add new_only
                        final_ids = new_only

            elif prefs.update_mode == "merge_dedupe":
                confirm = input(
                    "Merge mode will rebuild playlist ordering (existing first, then new uniques). Type MERGE to confirm: "
                ).strip()
                if confirm != "MERGE":
                    print("Cancelled.")
                    return
                merged = apply_update_mode(existing_ids, final_track_ids, "merge_dedupe")
                if pinned_ids:
                    merged = apply_pins_to_ids(pinned_ids, merged)
                clear_playlist(target_playlist)
                final_ids = merged
            else:
                final_ids = list(final_track_ids)

        if not final_ids:
            print("\nNothing to add/rebuild. Exiting.")
            return

        if not yn(
            f"\nProceed to update playlist '{getattr(target_playlist, 'name', '')}' with {len(final_ids)} track(s)?",
            default=True
        ):
            print("Cancelled.")
            return

        BATCH = 50
        for i in range(0, len(final_ids), BATCH):
            target_playlist.add(final_ids[i:i + BATCH])
            time.sleep(0.15)
            print(f"➕ Added {min(i + BATCH, len(final_ids))}/{len(final_ids)}")

        print("\n🎉 Done updating playlist.")
        print("Tip: open TIDAL on your device and download the playlist offline to your microSD.")
        return

    # CREATE new playlist
    if not yn("\nDo you want to CREATE a new playlist in your TIDAL account with the resolved tracks?", default=False):
        print("Cancelled.")
        return

    # Apply pins to create flow
    if pinned_ids:
        final_track_ids = apply_pins_to_ids(pinned_ids, final_track_ids)

    final_name = f"{playlist_name} ({time.strftime('%Y-%m-%d')})"
    playlist = session.user.create_playlist(final_name, description)
    print(f"\n✅ Playlist created: {playlist.name}\n")

    BATCH = 50
    for i in range(0, len(final_track_ids), BATCH):
        playlist.add(final_track_ids[i:i + BATCH])
        time.sleep(0.15)
        print(f"➕ Added {min(i + BATCH, len(final_track_ids))}/{len(final_track_ids)}")

    print("\n🎉 Done.")
    print("Tip: open TIDAL on your device and download the playlist offline to your microSD.")


if __name__ == "__main__":
    main()
