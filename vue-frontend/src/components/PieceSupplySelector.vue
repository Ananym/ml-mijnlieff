<template>
  <div
    class="piece-supply-selector"
    @wheel="handleScroll"
    @keydown="handleKeyDown"
    tabindex="0"
    ref="selectorRef"
  >
    <div
      v-for="type in Object.values(PieceTypes) as PieceType[]"
      :key="type"
      class="piece-row"
      @click="selectPiece(type)"
      :class="{
        selected: selectedPieceType === type,
        disabled: pieceCounts[isHumanFirst ? Players.ONE : Players.TWO][type] === 0 || disabled,
      }"
    >
      <div class="piece-icon">
        <component :is="getPieceIcon(type)" />
      </div>
      <div class="count-container">
        <span class="count human">{{
          pieceCounts[isHumanFirst ? Players.ONE : Players.TWO][type]
        }}</span>
        <span class="count ai">{{
          pieceCounts[isHumanFirst ? Players.TWO : Players.ONE][type]
        }}</span>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { ref, onMounted, PropType, Ref } from 'vue';
import { PieceTypes, Players, PieceType, PieceCounts } from '../gameLogic';
import DiagIcon from './icons/DiagIcon.vue';
import OrthoIcon from './icons/OrthoIcon.vue';
import NearIcon from './icons/NearIcon.vue';
import FarIcon from './icons/FarIcon.vue';

export default {
  name: 'PieceSupplySelector',
  components: {
    DiagIcon,
    OrthoIcon,
    NearIcon,
    FarIcon,
  },
  props: {
    pieceCounts: {
      type: Object as PropType<PieceCounts>,
      required: true,
    },
    isHumanFirst: {
      type: Boolean,
      required: true,
    },
    selectedPieceType: {
      type: Number,
      required: true,
    },
    disabled: {
      type: Boolean,
      default: false,
    },
  },
  emits: ['select-piece'],
  setup(props, { emit }) {
    const selectorRef: Ref<HTMLDivElement | null> = ref(null);

    const getPieceIcon = (pieceType: PieceType) => {
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

    const selectPiece = (type: PieceType) => {
      if (
        !props.disabled &&
        props.pieceCounts[props.isHumanFirst ? Players.ONE : Players.TWO][type] > 0
      ) {
        emit('select-piece', type);
      }
    };

    const cycleSelection = (direction: 1 | -1) => {
      // const types = Object.keys(props.pieceCounts).map(Number);
      // const currentIndex = types.indexOf(props.selectedPieceType);
      // let newIndex = (currentIndex + direction + types.length) % types.length;
      const player = props.isHumanFirst ? Players.ONE : Players.TWO;
      const selectedPieceType = props.selectedPieceType;
      for (let i = 0; i < 4; i++) {
        const proposedIndex = ((selectedPieceType + i * direction + 4) % 4) as PieceType;
        if (props.pieceCounts[player][proposedIndex] > 0) {
          selectPiece(proposedIndex);
          break;
        }
      }
    };

    const handleScroll = (event: WheelEvent) => {
      if (!props.disabled) {
        event.preventDefault();
        cycleSelection(event.deltaY > 0 ? 1 : -1);
      }
    };

    const handleKeyDown = (event: KeyboardEvent) => {
      if (!props.disabled && (event.key === 'ArrowUp' || event.key === 'ArrowDown')) {
        event.preventDefault();
        cycleSelection(event.key === 'ArrowDown' ? 1 : -1);
      }
    };

    onMounted(() => {
      if (selectorRef.value) {
        selectorRef.value.focus();
      }
    });

    // watch(
    //   () => props.selectedPieceType,
    //   (newType) => {
    //     if (props.pieceCounts[newType] === 0) {
    //       cycleSelection(1);
    //     }
    //   }
    // );

    return {
      getPieceIcon,
      selectPiece,
      handleScroll,
      handleKeyDown,
      selectorRef,
      Players,
      PieceTypes,
    };
  },
};
</script>

<style scoped>
.piece-supply-selector {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  margin-left: 20px;
  outline: none;
}

.piece-row {
  display: flex;
  align-items: center;
  margin: 5px 0px;
  cursor: pointer;
  padding: 5px;
  border-radius: 5px;
  transition: all 0.3s;
  width: 100%;
  border: 2px solid transparent;
  color: var(--text-color);
}

.piece-row:hover:not(.disabled) {
  background-color: var(--button-hover-bg);
}

.piece-row.selected:not(.disabled) {
  border: 2px solid var(--link-color);
}

.piece-row.disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.piece-icon {
  width: 40px;
  height: 40px;
  display: flex;
  justify-content: center;
  align-items: center;
  border: 2px solid transparent;
  border-radius: 5px;
  color: var(--text-color);
}

.count-container {
  display: flex;
  margin-left: 10px;
}

.count {
  font-weight: bold;
  padding: 2px 6px;
  border-radius: 3px;
  margin: 0 2px;
  color: black;
}

.human {
  background-color: #ccccff;
}

.ai {
  background-color: #ffcccc;
}

@media (prefers-color-scheme: dark) {
  .human {
    background-color: #8080ff;
  }

  .ai {
    background-color: #ff8080;
  }
}
</style>
