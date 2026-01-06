import numpy as np
import plotly.graph_objects as go


def apply_move(state, move):
    """
    Apply a single move to the cube state.
    The state is a numpy array of shape (6, 3, 3).
    Faces are indexed as:
        0: Up, 1: Left, 2: Front, 3: Right, 4: Back, 5: Down.
    Allowed moves: 'U', "U'", 'D', "D'", 'F', "F'", 'B', "B'", 'L', "L'", 'R', "R'"
    """
    new_state = state.copy()
    U, L, F, R, B, D = 0, 1, 2, 3, 4, 5
    old = state.copy()  # Preserve the original state for all read operations

    if move == "-":
        pass

    elif move == "U":
        new_state[U] = np.rot90(old[U], -1)
        new_state[F][0] = old[R][0].copy()
        new_state[R][0] = old[B][0].copy()
        new_state[B][0] = old[L][0].copy()
        new_state[L][0] = old[F][0].copy()

    elif move == "U'":
        new_state[U] = np.rot90(old[U], 1)
        new_state[F][0] = old[L][0].copy()
        new_state[L][0] = old[B][0].copy()
        new_state[B][0] = old[R][0].copy()
        new_state[R][0] = old[F][0].copy()

    elif move == "D":
        new_state[D] = np.rot90(old[D], -1)
        new_state[F][2] = old[L][2].copy()
        new_state[L][2] = old[B][2].copy()
        new_state[B][2] = old[R][2].copy()
        new_state[R][2] = old[F][2].copy()

    elif move == "D'":
        new_state[D] = np.rot90(old[D], 1)
        new_state[F][2] = old[R][2].copy()
        new_state[R][2] = old[B][2].copy()
        new_state[B][2] = old[L][2].copy()
        new_state[L][2] = old[F][2].copy()

    elif move == "F":
        new_state[F] = np.rot90(old[F], -1)
        new_state[U][2] = np.flip(old[L][:, 2].copy())
        new_state[L][:, 2] = old[D][0].copy()
        new_state[D][0] = np.flip(old[R][:, 0].copy())
        new_state[R][:, 0] = old[U][2].copy()

    elif move == "F'":
        new_state[F] = np.rot90(old[F], 1)
        new_state[U][2] = np.flip(old[R][:, 0].copy())
        new_state[R][:, 0] = np.flip(old[D][0].copy())
        new_state[D][0] = np.flip(old[L][:, 2].copy())
        new_state[L][:, 2] = np.flip(old[U][2].copy())

    elif move == "B":
        new_state[B] = np.rot90(old[B], -1)
        new_state[U][0] = old[R][:, 2].copy()
        new_state[R][:, 2] = np.flip(old[D][2].copy())
        new_state[D][2] = old[L][:, 0].copy()
        new_state[L][:, 0] = np.flip(old[U][0].copy())

    elif move == "B'":
        new_state[B] = np.rot90(old[B], 1)
        new_state[U][0] = np.flip(old[L][:, 0].copy())
        new_state[L][:, 0] = old[D][2].copy()
        new_state[D][2] = np.flip(old[R][:, 2].copy())
        new_state[R][:, 2] = old[U][0].copy()

    elif move == "L":
        new_state[L] = np.rot90(old[L], -1)
        new_state[U][:, 0] = old[B][:, 2].copy()
        new_state[B][:, 2] = np.flip(old[D][:, 0].copy())
        new_state[D][:, 0] = old[F][:, 0].copy()
        new_state[F][:, 0] = old[U][:, 0].copy()

    elif move == "L'":
        new_state[L] = np.rot90(old[L], 1)
        new_state[U][:, 0] = old[F][:, 0].copy()
        new_state[F][:, 0] = old[D][:, 0].copy()
        new_state[D][:, 0] = np.flip(old[B][:, 2].copy())
        new_state[B][:, 2] = old[U][:, 0].copy()

    elif move == "R":
        new_state[R] = np.rot90(old[R], -1)
        new_state[U][:, 2] = old[F][:, 2].copy()
        new_state[F][:, 2] = old[D][:, 2].copy()
        new_state[D][:, 2] = np.flip(old[B][:, 0].copy())
        new_state[B][:, 0] = np.flip(old[U][:, 2].copy())

    elif move == "R'":
        new_state[R] = np.rot90(old[R], 1)
        new_state[U][:, 2] = np.flip(old[B][:, 0].copy())
        new_state[B][:, 0] = np.flip(old[D][:, 2].copy())
        new_state[D][:, 2] = old[F][:, 2].copy()
        new_state[F][:, 2] = old[U][:, 2].copy()

    else:
        raise ValueError("Invalid move")
    return new_state


def apply_moves(state, moves):
    """
    Apply a sequence of moves to the cube state.
    """
    current_state = state.copy()  # Make a copy of the initial state
    for move in moves:
        current_state = apply_move(current_state, move)
    return current_state


def scramble_cube(state, n_moves=20):
    """
    Starting from the given cube state, apply n random moves to scramble it.

    Args:
        state: A numpy array of shape (6, 3, 3) representing the cube.
        n_moves: Number of random moves to apply.

    Returns:
        scrambled_state: A numpy array of shape (6, 3, 3) after scrambling.
        move_sequence: List of moves applied.
    """
    moves = ["-", "U", "U'", "D", "D'", "F", "F'", "B", "B'", "L", "L'", "R", "R'"]
    move_sequence = np.random.choice(moves, size=n_moves, replace=True)
    scrambled_state = state.copy()
    scrambled_state = apply_moves(scrambled_state, move_sequence)
    return scrambled_state, move_sequence


# Updated mapping: face labels correspond to the solved state's face indices:
# 0: Up -> yellow, 1: Left -> red, 2: Front -> green, 3: Right -> orange, 4: Back -> blue, 5: Down -> white.
color_map = {0: "yellow", 1: "red", 2: "green", 3: "orange", 4: "blue", 5: "white"}

# Define face parameters.
# We assume faces in order: 0: Up, 1: Left, 2: Front, 3: Right, 4: Back, 5: Down.
# The geometric positions remain the same.
face_params = {
    0: {
        "name": "Up",
        "center": np.array([0, 0, 1.5]),
        "u": np.array([-1, 0, 0]),
        "v": np.array([0, -1, 0]),
    },
    5: {
        "name": "Down",
        "center": np.array([0, 0, -1.5]),
        "u": np.array([-1, 0, 0]),
        "v": np.array([0, 1, 0]),
    },
    2: {
        "name": "Front",
        "center": np.array([0, 1.5, 0]),
        "u": np.array([-1, 0, 0]),
        "v": np.array([0, 0, 1]),
    },
    4: {
        "name": "Back",
        "center": np.array([0, -1.5, 0]),
        "u": np.array([1, 0, 0]),
        "v": np.array([0, 0, 1]),
    },
    1: {
        "name": "Left",
        "center": np.array([1.5, 0, 0]),
        "u": np.array([0, 1, 0]),
        "v": np.array([0, 0, 1]),
    },
    3: {
        "name": "Right",
        "center": np.array([-1.5, 0, 0]),
        "u": np.array([0, -1, 0]),
        "v": np.array([0, 0, 1]),
    },
}


def get_sticker_vertices(face_idx, i, j, face_params=face_params, sticker_size=1.0):
    """
    Compute the 3D coordinates of the four corners of a sticker on a given face.

    Args:
        face_idx (int): Index of the face (0 to 5).
        i, j (int): Row and column indices (0, 1, 2) where i=0 is the top row.
        face_params: Dictionary with face parameters.
        sticker_size: Size of each sticker.

    Returns:
        vertices: List of 4 vertices (each a 3D NumPy array).
    """
    params = face_params[face_idx]
    center = params["center"]
    u = params["u"]
    v = params["v"]

    # Full face is 3*sticker_size; half-length is 1.5*sticker_size.
    half_face = 1.5 * sticker_size

    # Calculate local coordinates:
    # Sticker centers are arranged on a 3x3 grid with centers at:
    # u_offset = -1 + j and v_offset = 1 - i.
    local_center = np.array([-1 + j, 1 - i]) * sticker_size
    # Offsets for the four corners (each sticker is 1 unit square).
    offsets = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
    vertices = []
    for du, dv in offsets:
        local = local_center + np.array([du, dv])
        vertex = center + local[0] * u + local[1] * v
        vertices.append(vertex)
    return vertices


def visualize_cube_state(state, face_params=face_params, color_map=color_map):
    """
    Visualize the Rubik's cube state using Plotly.

    Args:
        state: NumPy array of shape (6, 3, 3) representing the cube state.
        face_params: Dictionary with face parameters.
        color_map: Mapping from sticker label to color string.

    Returns:
        fig: A Plotly Figure object.
    """
    fig = go.Figure()

    # For each face and each sticker, compute the 3D polygon and add it as two triangles.
    for face in range(6):
        for i in range(3):
            for j in range(3):
                sticker_label = state[face, i, j]
                sticker_color = color_map[sticker_label]
                verts = get_sticker_vertices(face, i, j, face_params)
                # Triangulate the square: (v0, v1, v2) and (v0, v2, v3)
                triangles = [
                    [verts[0], verts[1], verts[2]],
                    [verts[0], verts[2], verts[3]],
                ]
                for tri in triangles:
                    x = [v[0] for v in tri]
                    y = [v[1] for v in tri]
                    z = [v[2] for v in tri]
                    fig.add_trace(
                        go.Mesh3d(
                            x=x,
                            y=y,
                            z=z,
                            color=sticker_color,
                            opacity=1.0,
                            flatshading=True,
                            showscale=False,
                            i=[0],
                            j=[1],
                            k=[2],
                            hoverinfo="skip",
                        )
                    )

                # Add black lines around the sticker
                for idx in range(4):
                    next_idx = (idx + 1) % 4
                    fig.add_trace(
                        go.Scatter3d(
                            x=[verts[idx][0], verts[next_idx][0]],
                            y=[verts[idx][1], verts[next_idx][1]],
                            z=[verts[idx][2], verts[next_idx][2]],
                            mode="lines",
                            line=dict(color="black", width=2),
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectratio=dict(x=1, y=1, z=1),
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
    )
    return fig
