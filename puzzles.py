from collections.abc import Sequence
import os

from absl import app
from absl import flags
import chess
import pandas as pd

from searchless_chess.src.engines import constants


_NUM_PUZZLES = flags.DEFINE_integer(
    name='num_puzzles',
    default=None,
    help='Number of puzzles to evaluate',
    required=True,
)

_AGENT = flags.DEFINE_enum(
    name='agent',
    default=None,
    enum_values=[
        'local',
        '9M',
        '136M',
        '270M',
        'stockfish',
    ],
    help='Agent to evaluate',
    required=True,
)

_MIN_RATING = flags.DEFINE_integer(
    name='min_rating',
    default=0,
    help='Minimum puzzle rating filter',
)


def solve_puzzle(board, moves, engine):
    """
    Returns True if engine solves FULL puzzle sequence.
    """
    for i in range(1, len(moves), 2):  # engine moves only
        predicted = engine.play(board=board).uci()
        expected = moves[i]

        if predicted != expected:
            return False

        board.push(chess.Move.from_uci(predicted))

        if i + 1 < len(moves):
            board.push(chess.Move.from_uci(moves[i + 1]))

    return True


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    puzzles_path = os.path.join(
        os.getcwd(),
        '../data/lichess_db_puzzle.csv',
    )

    puzzles = pd.read_csv(puzzles_path)

    # Optional rating filter
    if _MIN_RATING.value > 0:
        puzzles = puzzles[puzzles["Rating"] >= _MIN_RATING.value]

    # Sample puzzles
    puzzles = puzzles.sample(n=_NUM_PUZZLES.value, random_state=42)

    engine = constants.ENGINE_BUILDERS[_AGENT.value]()

    correct = 0
    total = 0

    for puzzle_id, puzzle in puzzles.iterrows():
        total += 1

        board = chess.Board(puzzle["FEN"])
        moves = puzzle["Moves"].split()

        # Apply opponent first move
        board.push(chess.Move.from_uci(moves[0]))

        # Copy board for prediction display
        display_board = board.copy()

        # Get expected + predicted first move
        expected_uci = moves[1]
        predicted_uci = engine.play(board=display_board).uci()

        try:
            expected_san = display_board.san(chess.Move.from_uci(expected_uci))
            predicted_san = display_board.san(chess.Move.from_uci(predicted_uci))
        except Exception:
            expected_san = expected_uci
            predicted_san = predicted_uci

        # Solve full puzzle
        solved = solve_puzzle(board.copy(), moves, engine)

        if solved:
            correct += 1

        # Pretty output
        print("\n==============================")
        print(f"Puzzle ID: {puzzle_id} | Rating: {puzzle['Rating']}")
        print(display_board.unicode())

        print("\nExpected :", expected_san)
        print("Predicted:", predicted_san)
        print("Solved   :", "✅" if solved else "❌")

    print("\n==============================")
    print(f"Accuracy: {correct}/{total} = {correct/total:.2%}")


if __name__ == '__main__':
    app.run(main)