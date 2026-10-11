# Frozen Output Fingerprints (Paper 1)

Fingerprint = (file size in bytes, first-12-chars of SHA1
over str(size) + first 4096 bytes + last 4096 bytes),
computed by src.io.sha1_head_tail. Recorded on the local
machine that produced these files. Hashes differ across
platforms (line endings, float formatting) â€” the ground
truth for any frozen-input check is the pair recorded here
for THIS machine.

## data/cleaned/ (frozen at end of Phase 4)

| City        | Size (bytes) | SHA1 head/tail |
|-------------|--------------|----------------|
| muntinlupa  |    6,896,272 | c95ee0e5944d   |
| baguio      |    6,884,091 | 391a9a50dd2d   |
| cebu        |    6,900,812 | 00e2fa338dc4   |
| davao       |    6,980,979 | 4e1d9372e0e9   |

## data/with_geometry/ (frozen at end of Phase 5)

| City        | Size (bytes) | SHA1 head/tail |
|-------------|--------------|----------------|
| muntinlupa  |    9,531,960 | 1825c0bdb60b   |
| baguio      |    9,527,013 | 340ec657074d   |
| cebu        |    9,533,960 | ceff379bb7c4   |
| davao       |    9,603,594 | fcfb31efe3d3   |

## Policy

- data/cleaned/*.csv is FROZEN. Never modify.
- data/with_geometry/*.csv is FROZEN. Never modify.
- Any future phase that needs to update either must do so in
  a new folder, and the fingerprint here must be updated in
  the same commit.
- Every downstream script that reads from a frozen folder
  should snapshot the fingerprints at start and end of run,
  and fail loudly if they change.

## Phase 7 â€” Train/Val/Test Split

Frozen at: 2026-10-07 11:57 UTC
Script: scripts/06_split.py
Fingerprints taken BEFORE and AFTER the split; identical, so the frozen Phase 6 features files are unchanged by this stage.

| file                                       | size (bytes) | sha1_head_tail |
|--------------------------------------------|--------------|----------------|
| muntinlupa.csv                             |   25,634,740 | 2cc9ebaf82c41fa7a20d18f9cc02c2228ca5d6ad |
| baguio.csv                                 |   25,640,263 | 7fb600a9a34b66eccb1ceb6c78ff137573801ec9 |
| cebu.csv                                   |   25,586,962 | f673164fb7a1e76c721b74831afd1002056d3c3d |
| davao.csv                                  |   25,815,960 | 2f249ff64a1a9d9cd4588cb95e6a67387b037eb9 |

| file                                       | size (bytes) | sha1_head_tail |
|--------------------------------------------|--------------|----------------|
| muntinlupa_train.csv                       |   12,588,527 | 73396369200341d2f3a635e7aa8de4ffda0088fb |
| muntinlupa_val.csv                         |    4,229,311 | a316bb5578772cf547ee35ab559169c4cece3f49 |
| muntinlupa_test.csv                        |    8,407,845 | 707717ae51be6e4c1d96233a7286f79784d2643f |
| baguio_train.csv                           |   12,591,738 | 4bc2ca15d9a22058e90f6a36ca40b14bd18031f1 |
| baguio_val.csv                             |    4,231,033 | bcfecb9e3e4e37e09990aaa5cf5405f04bf91f5c |
| baguio_test.csv                            |    8,410,626 | 41e09b250fd1ffed77cde8ad5ec45da299123409 |
| cebu_train.csv                             |   12,572,854 | 0da1fa09866128372a5c252c2ee51146b1e918c4 |
| cebu_val.csv                               |    4,211,621 | ee339cfd17101b1d759b3e11b7cdd4f4f5824b61 |
| cebu_test.csv                              |    8,412,031 | 8b5ab5c5c9abaaaaf215dacfaa95c7e657cc7f2c |
| davao_train.csv                            |   12,643,700 | 7d51cbfca898bc3150b41e61fab4b011af89a14f |
| davao_val.csv                              |    4,278,855 | d70779d14c6e45f21cfeb4440112649415577d91 |
| davao_test.csv                             |    8,462,460 | 1911fcb94a690043a2aef067d50d97fc430c5023 |

Boundary note: paper1_config.py stores inclusive END dates (e.g. TRAIN_END = '2024-06-30' means the last included day). This script converts each to an exclusive upper bound (END + 1 day at 00:00 UTC) before masking, yielding the half-open intervals:
  train: [2023-01-01 00:00 UTC, 2024-07-01 00:00 UTC)
  val:   [2024-07-01 00:00 UTC, 2025-01-01 00:00 UTC)
  test:  [2025-01-01 00:00 UTC, 2026-01-01 00:00 UTC)
No shuffling, no sklearn train_test_split. The warmup rows (first 168 per city, zero-filled lag_168) are contained entirely within the train split.
