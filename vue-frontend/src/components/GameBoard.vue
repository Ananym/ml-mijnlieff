<template>
  <div class="game-board" :class="{ disabled: isAIThinking }">
    <div v-for="(column, columnIndex) in board" :key="columnIndex" class="board-column">
      <div
        v-for="(cell, rowIndex) in column"
        :key="rowIndex"
        class="board-cell"
        :class="{
          'human-player': isHumanCell(cell),
          'ai-player': isAICell(cell),
          'valid-move': !isAIThinking && isValidMove(columnIndex, rowIndex),
          'ai-valid-move': isAIThinking && isValidMove(columnIndex, rowIndex),
          'last-ai-move': isLastAIMove(columnIndex, rowIndex),
        }"
        @click="makeMove(columnIndex, rowIndex)"
      >
        <svg v-if="cell[0] !== 0 || cell[1] !== 0" width="40" height="40" viewBox="0 0 40 40">
          <component :is="getPieceIcon(columnIndex, rowIndex)" />
        </svg>
        <!-- <div style="font-size: 16px;">{{ columnIndex}},{{ rowIndex }} </div> -->
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { PropType } from 'vue';
import { Move, Board, PieceTypes, Cell } from '../gameLogic';
import DiagIcon from './icons/DiagIcon.vue';
import OrthoIcon from './icons/OrthoIcon.vue';
import NearIcon from './icons/NearIcon.vue';
import FarIcon from './icons/FarIcon.vue';

export default {
  name: 'GameBoard',
  components: {
    DiagIcon,
    OrthoIcon,
    NearIcon,
    FarIcon,
  },
  props: {
    board: {
      type: Array as PropType<Board>,
      required: true,
    },
    validMoves: {
      type: Array as PropType<Move[]>,
      required: true,
    },
    isAIThinking: {
      type: Boolean,
      required: true,
    },
    lastAIMove: {
      type: Object as PropType<Move | null>,
      required: false,
      default: null,
    },
    selectedPieceType: {
      type: Number,
      required: true,
    },
    isHumanFirst: {
      type: Boolean,
      required: true,
    },
  },
  setup(props, { emit }) {
    const isValidMove = (x: number, y: number) => {
      return props.validMoves.some(
        (move: Move) => move.x === x && move.y === y && move.pieceType === props.selectedPieceType
      );
    };

    const makeMove = (x: number, y: number) => {
      if (!props.isAIThinking && isValidMove(x, y)) {
        emit('make-move', x, y);
      }
    };

    const getPieceIcon = (x: number, y: number) => {
      const cell = props.board[x][y];
      const pieceType = cell[0] > 0 ? cell[0] : cell[1];
      switch (pieceType) {
        case PieceTypes.DIAG:
          return DiagIcon;
        case PieceTypes.ORTHO:
          return OrthoIcon;
        case PieceTypes.NEAR:
          return NearIcon;
        case PieceTypes.FAR:
          return FarIcon;
        default:
          return null;
      }
    };

    const isLastAIMove = (x: number, y: number) => {
      return props.lastAIMove && props.lastAIMove.x === x && props.lastAIMove.y === y;
    };

    const isHumanCell = (cell: Cell) => {
      if (props.isHumanFirst) {
        return cell[0] > 0;
      } else {
        return cell[1] > 0;
      }
    };

    const isAICell = (cell: Cell) => {
      if (props.isHumanFirst) {
        return cell[1] > 0;
      } else {
        return cell[0] > 0;
      }
    };

    return {
      isValidMove,
      makeMove,
      getPieceIcon,
      isLastAIMove,
      isHumanCell,
      isAICell,
    };
  },
};
</script>

<style scoped>
.game-board {
  display: flex;
  border: 2px solid #333;
  /* flex-direction: column; */
}

.game-board.disabled {
  opacity: 0.7;
  pointer-events: none;
}

.board-column {
  display: flex;
  flex-direction: column;
}

.board-cell {
  width: 60px;
  height: 60px;
  border: 1px solid #ccc;
  display: flex;
  justify-content: center;
  align-items: center;
  cursor: pointer;
}

.human-player {
  background-color: #ccccff; /* Blue */
}

.ai-player {
  background-color: #ffcccc; /* Red */
}

.valid-move {
  background-color: #ccffcc;
}

.ai-valid-move {
  background-color: #fffecc;
}

.last-ai-move {
  box-shadow: inset 0 0 10px #ff0000;
}
</style>
