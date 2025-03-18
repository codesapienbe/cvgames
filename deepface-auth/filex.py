import os

def get_local_dirs():
        
    base_dir = os.path.join(os.path.dirname(__file__), "data")
    print(f"Base directory set to: {base_dir}")

    events_dir = os.path.join(base_dir, "events")
    print(f"Events directory set to: {events_dir}")

    identities_dir = os.path.join(base_dir, "identities")
    print(f"Identities directory set to: {identities_dir}")

    checksums_dir = os.path.join(base_dir, "checksums")
    print(f"Checksums directory set to: {checksums_dir}")

    faces_dir = os.path.join(base_dir, "faces")
    print(f"Faces directory set to: {faces_dir}")

    audios_dir = os.path.join(base_dir, "audios")
    print(f"Audios directory set to: {audios_dir}")

    lookups_dir = os.path.join(base_dir, "lookups")
    print(f"Lookups directory set to: {lookups_dir}")

    return base_dir, events_dir, identities_dir, checksums_dir, faces_dir, audios_dir, lookups_dir