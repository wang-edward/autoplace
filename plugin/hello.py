from kipy import KiCad
from kipy.geometry import Vector2, Angle

if __name__ == "__main__":
    try:
        kicad = KiCad()
        print(f"Connected to KiCad {kicad.get_version()}")
    except BaseException as e:
        print(f"Not connected to KiCad: {e}")

    board = kicad.get_board()
    footprints = board.get_footprints()

    for f in footprints:
        print(f"{f.reference_field.text.value}")
