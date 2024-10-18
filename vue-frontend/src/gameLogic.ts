export class IllegalMoveException extends Error {
  constructor(player: Player, move: Move) {
    super(
      move !== undefined ?
      `Player ${player} tried to move to (${move.x}, ${move.y}) with piece type ${move.pieceType}, illegal move!`
      :
      `Player ${player} tried to make an undefined move!`
    );
  }
}

export const Players = {
  ONE: 0,
  TWO: 1,
} as const;

export type Player = (typeof Players)[keyof typeof Players];

export const PieceTypes = {
  DIAG: 1,
  ORTHO: 2,
  NEAR: 3,
  FAR: 4,
} as const;

export type PieceType = (typeof PieceTypes)[keyof typeof PieceTypes];

export const PieceNames: Record<PieceType, string> = {
  1: 'Diagonal',
  2: 'Orthogonal',
  3: 'Near',
  4: 'Far',
} as const;

export const TurnResults = {
  NORMAL: 0,
  OPP_MUST_PASS: 1,
  GAME_OVER: 2,
} as const;

export type TurnResult = (typeof TurnResults)[keyof typeof TurnResults];

export const GameTypes = {
  AI_VS_AI: 0,
  AI_VS_HUMAN: 1,
  HUMAN_VS_HUMAN: 2,
} as const;

export type GameType = (typeof GameTypes)[keyof typeof GameTypes];

export type Move = {
  x: number;
  y: number;
  pieceType: PieceType;
};

export type Cell = [number, number];
export type Board = Cell[][];
export type PieceCounts = {
  [P in Player]: {
    [T in PieceType]: number;
  };
};
type LastMove = Move | null;
export type GameState = {
  board: Board;
  currentPlayer: Player;
  pieceCounts: PieceCounts;
  lastMove: LastMove;
  gameType: GameType;
};

export function moveToString(move: Move, player: Player): string {
  return `Player ${player} played (${move.x}, ${move.y}, ${PieceNames[move.pieceType]})`;
}

export function initializeGameState(gameType: GameType): GameState {
  if (gameType !== GameTypes.AI_VS_HUMAN) {
    throw new Error('Unsupported');
  }
  return {
    board: Array(4)
      .fill(null)
      .map(() =>
        Array(4)
          .fill(null)
          .map(() => [0, 0])
      ),
    currentPlayer: Players.ONE,
    pieceCounts: {
      [Players.ONE]: {
        [PieceTypes.DIAG]: 2,
        [PieceTypes.ORTHO]: 2,
        [PieceTypes.NEAR]: 2,
        [PieceTypes.FAR]: 2,
      },
      [Players.TWO]: {
        [PieceTypes.DIAG]: 2,
        [PieceTypes.ORTHO]: 2,
        [PieceTypes.NEAR]: 2,
        [PieceTypes.FAR]: 2,
      },
    },
    lastMove: null,
    gameType,
  };
}

export function getValidMoves(gameState: GameState): Move[] {
  const validMoves: Move[] = [];
  for (let x = 0; x < 4; x++) {
    for (let y = 0; y < 4; y++) {
      for (const pieceType of Object.values(PieceTypes)) {
        if (isValidMove(gameState, { x, y, pieceType })) {
          validMoves.push({ x, y, pieceType });
        }
      }
    }
  }
  return validMoves;
}

function switchPlayer(gameState: GameState) {
  gameState.currentPlayer = gameState.currentPlayer === Players.ONE ? Players.TWO : Players.ONE;
  // console.log(
  //   `Current player switched to ${
  //     gameState.currentPlayer === Players.ONE ? 'Player One' : 'Player Two'
  //   }`
  // );
}

export function currentPlayerHasPieces(gameState: GameState): boolean {
  return Object.values(gameState.pieceCounts[gameState.currentPlayer]).some((count) => count !== 0);
}

export function currentPlayerHasValidMoves(gameState: GameState) {
  return getValidMoves(gameState).length > 0;
}

export function makeMove(gameState: GameState, move: Move): TurnResult {
  if (!isValidMove(gameState, move) || move===undefined) {
    console.log(`Invalid move: ${move}`);
    throw new IllegalMoveException(gameState.currentPlayer, move);
  }
  const { x, y, pieceType } = move;
  console.log(`Processing move ${moveToString(move, gameState.currentPlayer)}`);
  gameState.board[x][y][gameState.currentPlayer] = pieceType;

  gameState.pieceCounts[gameState.currentPlayer][pieceType]--;
  gameState.lastMove = { x, y, pieceType };

  const selfHasPieces = currentPlayerHasPieces(gameState);
  switchPlayer(gameState);
  const oppHasPieces = currentPlayerHasPieces(gameState);
  const oppMustPass = !currentPlayerHasValidMoves(gameState);

  if (oppMustPass && !selfHasPieces) {
    console.log(
      'Game over because opponent has zero valid moves and current player has zero pieces'
    );
    return TurnResults.GAME_OVER;
  } else if (!oppHasPieces) {
    console.log('Game over because opponent has zero pieces');
    return TurnResults.GAME_OVER;
  } else if (oppMustPass) {
    console.log('Opponent must pass because opp has zero valid moves');
    return TurnResults.OPP_MUST_PASS;
  } else {
    return TurnResults.NORMAL;
  }
}

export function isValidMove(gameState: GameState, move: Move): boolean {

  const { x, y, pieceType } = move;

  if (gameState.board[x][y].some((piece) => piece !== 0)) {
    return false;
  }

  if (gameState.pieceCounts[gameState.currentPlayer][pieceType] === 0) {
    return false;
  }

  if (boardIsEmpty(gameState)) {
    // Only allow moves on the outer edge of the board
    return x === 0 || x === 3 || y === 0 || y === 3;
  }

  if (gameState.lastMove === null) {
    // cell is empty, not first turn, has piece type remaining, no restrictions apply
    return true;
  }

  const { x: lastX, y: lastY, pieceType: lastPieceType } = gameState.lastMove;

  switch (lastPieceType) {
    case PieceTypes.DIAG:
      return Math.abs(x - lastX) === Math.abs(y - lastY) && y !== lastY;
    case PieceTypes.ORTHO:
      // we already check for cell empty
      return x === lastX || y === lastY;
    case PieceTypes.NEAR:
      return Math.abs(y - lastY) <= 1 && Math.abs(x - lastX) <= 1 && (y !== lastY || x !== lastX);
    case PieceTypes.FAR:
      return Math.abs(y - lastY) > 1 || Math.abs(x - lastX) > 1;
    default:
      return false;
  }
}

function boardIsEmpty(gameState: GameState): boolean {
  return gameState.board.every((plane) => plane.every((row) => row.every((value) => value === 0)));
}

export function calculateScores(gameState: GameState): Record<Player, number> {
  let scores: Record<Player, number> = {
    [Players.ONE]: 0,
    [Players.TWO]: 0,
  };

  function get_val_or_none(x: number, y: number, player: Player) {
    if (x < 0 || x >= 4 || y < 0 || y >= 4) {
      return null;
    }
    return gameState.board[x][y][player];
  }

  for (const player of [Players.ONE, Players.TWO]) {
    for (let x = 0; x < 4; x++) {
      for (let y = 0; y < 4; y++) {
        for (const delta of [
          { x: 1, y: 0 },
          { x: 0, y: 1 },
          { x: 1, y: 1 },
          { x:-1, y: 1 },
        ]) {
          const previousCellCoords = { x: x - delta.x, y: y - delta.y };
          if (get_val_or_none(previousCellCoords.x, previousCellCoords.y, player)) {
            // previous cell in line is occupied, so this is part of an existing line, skip
            continue;
          }
          let length = 0;
          while (true) {
            const currentCellCoords = {
              x: x + delta.x * length,
              y: y + delta.y * length,
            };
            const currentCell = get_val_or_none(currentCellCoords.x, currentCellCoords.y, player);
            if (currentCell === null) {
              break;
            }
            if (currentCell === 0) {
              break;
            }
            length++;
          }
          if (length == 4) {
            scores[player] += 2;
          } else if (length == 3) {
            scores[player] += 1;
          }
        }
      }
    }
  }

  return scores;
}

export function passTurn(gameState: GameState) {
  switchPlayer(gameState);
  gameState.lastMove = null;
}

export function getStateRepresentation(gameState: GameState) {
  return {
    board: gameState.board,
    currentPlayer: gameState.currentPlayer,
    pieceCounts: gameState.pieceCounts,
    lastMove: gameState.lastMove,
    difficulty: 1,
  };
}

export function getDebugStringRepresentation(gameState: GameState) {
  const cellToString = (cell: Cell) => {
    if (cell.every((value) => value === 0)) {
      return '__';
    }
    if (cell[0] > 0) {
      return `A${cell[0]}`;
    } else {
      return `B${cell[1]}`;
    }
  };

  let stringBoardRep = '';
  for (let y = 0; y < 4; y++) {
    for (let x = 0; x < 4; x++) {
      stringBoardRep += cellToString(gameState.board[x][y]);
      stringBoardRep += ', ';
    }
    stringBoardRep += '\n';
  }

  return `Current Player: ${gameState.currentPlayer}\nBoard:\n${stringBoardRep}\n`;
}
