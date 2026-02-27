MAP_W: int = 1181
MAP_H: int = 1133

# Piksel başına yaklaşık 0.1476 m/px oranıyla kalibre edilmiş değerler
PHASE_RADIUS: dict[int, int] = {
    1: 298,
    2: 162,
    3: 99,
    4: 48,
}

_R1 = PHASE_RADIUS[1]
PHASE1_X_MIN: int = _R1
PHASE1_X_MAX: int = MAP_W - _R1
PHASE1_Y_MIN: int = _R1
PHASE1_Y_MAX: int = MAP_H - _R1
