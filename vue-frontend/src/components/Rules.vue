<template>
  <div v-if="show" class="modal-overlay" @click="closeModal">
    <div class="modal-content" @click.stop>
      <h2>Game Rules</h2>
      <div class="rules-text">
        <p>
          Players take alternating turns on a 4x4 grid, trying to form lines of three or four
          pieces.
        </p>
        <p>Players start with a supply of two of each of the four piece types.</p>
        <p>Each turn, a player places a piece from their supply on an empty cell.</p>
        <p>Each piece type enforces rules on where the opponent can play on their next turn.</p>
        <h3>Piece Types and Placement Rules:</h3>
        <ul>
          <li>
            <div class="piece-rule">
              <div class="piece-icon"><DiagIcon /></div>
              <strong>Diag:</strong> Forces opponent to play diagonally at any distance.
            </div>
          </li>
          <li>
            <div class="piece-rule">
              <div class="piece-icon"><OrthoIcon /></div>
              <strong>Ortho:</strong> Forces opponent to play orthogonally at any distance.
            </div>
          </li>
          <li>
            <div class="piece-rule">
              <div class="piece-icon"><NearIcon /></div>
              <strong>Near:</strong> Forces opponent to play adjacent (diagonally or orthogonally).
            </div>
          </li>
          <li>
            <div class="piece-rule">
              <div class="piece-icon"><FarIcon /></div>
              <strong>Far:</strong> Forces opponent to play at least two cells away (diagonally or
              orthogonally).
            </div>
          </li>
        </ul>
        <p>
          If a player cannot make a legal move, they pass their turn. The opponent's next turn has
          no restrictions.
        </p>
        <p>The game ends when a player starts their turn with zero pieces remaining.</p>
        <p>An opponent can make one last move after a player plays their final piece.</p>
        <h3>Scoring:</h3>
        <ul>
          <li>A line of three: 1 point</li>
          <li>A line of four: 2 points</li>
        </ul>
        <p>The player with the most points wins. Draws are possible.</p>
      </div>
      <button @click="closeModal" class="close-button">Close</button>
    </div>
  </div>
</template>

<script lang="ts">
import DiagIcon from './icons/DiagIcon.vue';
import OrthoIcon from './icons/OrthoIcon.vue';
import NearIcon from './icons/NearIcon.vue';
import FarIcon from './icons/FarIcon.vue';

export default {
  name: 'Rules',
  components: {
    DiagIcon,
    OrthoIcon,
    NearIcon,
    FarIcon,
  },
  props: {
    show: Boolean,
  },
  emits: ['close'],
  setup(props, { emit }) {
    const closeModal = () => {
      emit('close');
    };

    return {
      closeModal,
    };
  },
};
</script>

<style scoped>
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
}

.modal-content {
  background-color: var(--bg-color);
  color: var(--text-color);
  padding: 20px;
  border-radius: 5px;
  max-width: 80%;
  max-height: 80%;
  overflow-y: auto;
  border: 1px solid var(--button-border);
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

h2,
h3 {
  text-align: center;
  margin-bottom: 20px;
  color: var(--text-color);
}

.rules-text {
  text-align: left;
  color: var(--text-color);
}

ul {
  padding-left: 20px;
  color: var(--text-color);
}

.close-button {
  display: block;
  margin: 20px auto 0;
  padding: 10px 20px;
  font-size: 16px;
  cursor: pointer;
  background-color: var(--button-bg);
  border: 2px solid var(--button-border);
  color: var(--text-color);
  border-radius: 5px;
  transition: all 0.2s;
}

.close-button:hover {
  background-color: var(--button-hover-bg);
}

.piece-rule {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 10px 0;
}

.piece-icon {
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-color);
}

.piece-icon :deep(svg) {
  width: 100%;
  height: 100%;
  color: inherit;
}

strong {
  color: var(--link-color);
}
</style>
