# All years are served via the CKAN Datastore API. Resource IDs are stable
# even when the portal renames or replaces the underlying file.
#
# BUOYS is keyed by short slug; each value maps year → CKAN resource ID.
# Only resources whose native cadence is the project's 30-minute grid are
# included — the deep historical bundles for Mooloolaba (2000-2014, 1 h),
# Brisbane (1976-2011, 12 h), Gold Coast (1987-2014, 6 h), and Tweed Heads
# (1995-2011, 1 h) are intentionally excluded. The North Moreton Bay
# 2010-2015 bundle and the Caloundra 2013-2015 bundle are 30 min and stay.
# Coverage by buoy:
#   mooloolaba         — 2015-onwards
#   caloundra          — 2013-onwards (2013-2015 bundle + per-year 2016+)
#   brisbane           — 2012-onwards (per-year 2012/13/14 + 2015+)
#   north-moreton-bay  — 2010-onwards (2010-2015 bundle + per-year 2016+)
#   gold-coast         — 2014-onwards (per-year 2014 + 2015+)
#   tweed-heads        — 2012-onwards (per-year 2012/13/14 + 2015+)
#   palm-beach         — 2017-onwards (deployed mid-2017)
#   wide-bay           — 2019-onwards (deployed 2019)
BUOYS: dict[str, dict[int, str]] = {
    "mooloolaba": {
        2015: "81df149b-67fc-4e5c-8ab8-b479001e04eb",
        2016: "8e7fcf48-ac17-45ea-b4f3-b30cd1739658",
        2017: "61faa02e-a4e5-400e-98ff-3266b255da1b",
        2018: "b92629f7-a79f-45a2-857d-2217b2f11e63",
        2019: "d80243cd-5bca-41a0-b0f8-4d4fac168d94",
        2020: "f58445e2-44cc-48e6-ad99-1ee56e1ee402",
        2021: "b700d6d6-31fa-43b1-ae89-1fa811765aff",
        2022: "c995189b-82ec-422b-bf56-ccc5510618bc",
        2023: "fbad2855-d2cb-4889-ba1e-bee2c042d1c2",
        2024: "51c8f862-5ecd-4cff-af35-968ddd48a16e",
        2025: "c0cfd5c8-59c1-4fcf-b9d9-75bce4149808",
    },
    "caloundra": {
        # 2013 resource is a bundle covering 2013-2015; pre-2017 column schema.
        2013: "fe51b69b-df4e-4eb2-981f-3e18707bd09f",
        2016: "be62134c-15a2-4736-8b2a-bbfd323f10ce",
        2017: "80481a30-ac2d-4e67-82e9-d5fcb282f58b",
        2018: "e302c519-77ec-46e5-b819-767ac177f37a",
        2019: "6603ad4e-98cd-4215-877c-61b9759cbbd0",
        2020: "fdd1bc53-1d59-46bd-a5a6-90c18f851bd8",
        2021: "43354fd8-4a9b-4004-b42f-2fcb9e8d5b8a",
        2022: "a82ce37b-cca5-44db-a2f1-14c2b6cea836",
        2023: "f7a3f62e-4021-4d39-9fdd-4e13b5d62223",
        2024: "a7d465ba-0cab-4b4e-a93c-4c8c86c196b5",
        2025: "d276f08f-5853-493e-a607-0dfa60f6e850",
    },
    "brisbane": {
        # 2012-2014 are per-year 30-minute files; the 1976-2011 bundle is
        # 12-hour cadence and is therefore excluded.
        2012: "cf594247-4da2-4802-8a86-ea7b514df3e7",
        2013: "5e648c66-e67f-4b76-9ba3-5a281bec7ccf",
        2014: "b328dc90-4f16-4c63-a577-ad954b5e898c",
        2015: "53cfe709-5b7f-4339-b8c7-919cdcdb79ae",
        2016: "19b441dd-2539-497b-b11c-78a85def64c9",
        2017: "3a833ec3-2685-4999-af4f-f10d304042f6",
        2018: "de0ddbda-84e5-4583-a85b-bea48bd875d5",
        2019: "de65743a-daf9-4726-a0ee-fb6c2ea641b4",
        2020: "0e525f71-8df4-4cf2-8f57-76f66b7b67c8",
        2021: "16b71762-862f-4cd6-909b-ff83de7ec144",
        2022: "4f38dc47-11b9-422b-93ba-82913e972b36",
        2023: "4bff0f4e-9739-45b7-a92e-bc640cbc9bba",
        2024: "7f7da919-6c68-4bf4-bb4b-b6a0d936316c",
        2025: "60abceb3-2949-48df-8181-3f98ae72108d",
    },
    "north-moreton-bay": {
        # 2010 resource is a bundle covering 2010-2015; pre-2017 column schema.
        2010: "6ab26f13-ee41-4ef1-ba22-846aaeaaee6f",
        2016: "4c24fddd-af86-41ea-8a03-9c33941f8f10",
        2017: "32838f8b-496a-4056-9c3f-b6a52932f246",
        2018: "b1386a01-e4d5-462f-bf84-a77e519f0cc6",
        2019: "b9fda7ae-c24e-4bb9-b2f4-83487327300d",
        2020: "cbcd6887-b9f6-40cb-9fc4-8e313fbd4e53",
        2021: "495628d7-1634-4a6a-9921-58078eaffe2c",
        2022: "c7dfd4b3-7365-4ecb-ad6e-93b396df9e94",
        2023: "eb6e9ea6-72be-4148-b924-5b10f65ef201",
        2024: "48682394-8098-4ab6-8e6a-d2da453a220b",
        2025: "95654258-beee-4b4e-b506-e49b6657e5dd",
    },
    "gold-coast": {
        # 2014 is a per-year 30-minute file; the 1987-2014 historical bundle
        # is 6-hour cadence and is therefore excluded.
        2014: "9c1270fd-6626-4ed4-b629-73f687dbb04f",
        2015: "30cdfd68-52e9-4c5c-933c-03c2fed5a11a",
        2016: "c5e598d5-5a9a-45be-9da6-a9f47042e006",
        2017: "f85931b4-926a-49e3-9e56-d65bd49a9f14",
        2018: "d1049f97-45a9-4b3d-80be-7fce8cd3ec29",
        2019: "ee5859f9-e55c-434f-b20c-7da30b1e53e1",
        2020: "73b4e42a-f3e7-4632-8b8e-d52205899048",
        2021: "edc414a8-3ffa-47a2-9d9b-14f07eb22072",
        # 2022 Mk4 (e0068000-…) skipped — empty Date/Time fields in datastore.
        2022: "2eeb6b4d-f52a-45c1-b640-336aaf53b40b",
        2023: "618d4d1e-fa39-4e04-929a-94ca6e107973",
        2024: "67c8cf49-cb29-4cd1-86ff-d61bfdb8cbba",
        2025: "a8a12129-c99d-45f6-832b-a5cee4754b54",
    },
    "tweed-heads": {
        # Furthest-south SE QLD buoy; jointly operated with NSW. 2012-2014 are
        # per-year 30-minute files; the 1995-2011 bundle is 1-hour cadence and
        # is therefore excluded.
        2012: "4ca6d884-f135-44bb-9c0e-a437fdafd247",
        2013: "5d269a88-afec-4aeb-88c9-903d924cf30b",
        2014: "34c769b5-8a14-49f7-ae43-eaba453875f4",
        2015: "0a6087f2-d917-4009-b4d1-1b0b91fa9a87",
        2016: "74341a48-933c-4b64-b496-59dde6ac5e57",
        2017: "8d6b1736-7ae2-484b-bf75-a7ef0564da63",
        2018: "34860155-b168-4b81-b48f-dc362b9807bd",
        2019: "7caf54dd-356e-4d9d-862d-f0fd86cad69e",
        2020: "25bf55cc-4c7b-4ca6-9f07-e2f197314b02",
        2021: "68005deb-5f79-4808-bc16-3b5378eda85a",
        2022: "14eb287f-8084-42a9-a4e0-035db5ff283b",
        2023: "45968dc5-40f4-46e0-9c8f-915947f879b3",
        2024: "7f6c61b7-5a19-48c9-a609-1d1bed8b73fe",
        2025: "bc667055-aa5c-4449-ba26-3953e99ccf67",
    },
    "palm-beach": {
        # Inshore Gold Coast site ~25 km south of the gold-coast buoy; deployed
        # 2017, so 2015-2016 are unavailable.
        2017: "4da6606f-cd31-4e55-8e28-058e86e5d8c4",
        2018: "8b238223-0936-4e67-985f-f747f0b8933c",
        2019: "fd2898c4-cbf1-490f-a417-524cd1aa966c",
        2020: "f9428896-ed80-4a7c-a94c-f8ba1e305cae",
        2021: "524a0e1d-8a05-4d04-abfd-6d45f29538ad",
        2022: "8351f685-066f-4a75-80f6-cbd71decd3e9",
        2023: "c5df48a2-9c02-46c9-bfe3-d4df4b5d7375",
        2024: "e290a20a-7a2c-4004-bbb7-a379ff12bb02",
        2025: "6fa75109-b4b5-4360-a3e4-ed6dfd80efc3",
    },
    "wide-bay": {
        # Northern buoy ~120 km north of Mooloolaba; the only buoy upstream of
        # northerly swells. Deployed 2019, so 2015-2018 are unavailable.
        2019: "ac458084-beb6-4845-92e5-06b35bce32c4",
        2020: "7880277a-abe1-46f7-b157-06b95d5bcbcf",
        2021: "a6d74800-9c8a-4d0e-826b-b139d0d8bfe5",
        2022: "1ce31773-1031-4deb-81ac-70f7d3f23111",
        2023: "1bf8a850-dd59-441b-b0dd-11acec002227",
        2024: "327b3beb-ed49-44eb-9146-7c335962f17c",
        2025: "ae855051-21d5-4913-9797-04645fd7abf6",
    },
}

# Backwards-compatible alias for the primary Mooloolaba buoy.
RESOURCE_IDS: dict[int, str] = BUOYS["mooloolaba"]

# Raw source files use -99.9 to indicate missing/erroneous readings
SENTINEL_VALUE = -99.9

# Final standardised column names applied after per-year normalisation
COLUMN_RENAME_MAP: dict[str, str] = {
    "Date/Time": "datetime",
    "Hs": "hsig_m",
    "Hmax": "hmax_m",
    "Tz": "tz_s",
    "Tp": "tp_s",
    "Peak Direction": "peak_dir_deg",
    "SST": "sst_c",
}
