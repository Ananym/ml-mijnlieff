<template>
  <div v-if="show" class="modal-overlay" @click="closeModal">
    <div class="modal-content" @click.stop>
      <h2>Game Rules</h2>
      <div class="rules-text">
        <p>Players take alternating turns on a 4x4 grid.</p>
        <p>On each turn, a player places a piece on an empty cell.</p>
        <p>Initial supply per player: 2x Diag, 2x Ortho, 2x Near, 2x Far.</p>
        <p>Pieces are taken from the player's supply when played.</p>
        <p>Players cannot use a piece type if none remain in their supply.</p>
        <h3>Piece Types and Placement Rules:</h3>
        <ul>
          <li>
            <strong>Diag:</strong> Forces opponent to play diagonally at any
            distance.
          </li>
          <li>
            <strong>Ortho:</strong> Forces opponent to play orthogonally at any
            distance.
          </li>
          <li>
            <strong>Near:</strong> Forces opponent to play adjacent (diagonally
            or orthogonally).
          </li>
          <li>
            <strong>Far:</strong> Forces opponent to play at least two cells
            away (diagonally or orthogonally).
          </li>
        </ul>
        <p>
          If a player cannot make a legal move, they pass their turn. The
          opponent's next turn has no restrictions.
        </p>
        <p>
          The game ends when a player starts their turn with zero pieces
          remaining.
        </p>
        <p>
          An opponent can make one last move after a player plays their final
          piece.
        </p>
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
export default {
  name: "Rules",
  props: {
    show: Boolean,
  },
  emits: ["close"],
  setup(props, { emit }) {
    const closeModal = () => {
      emit("close");
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
  background-color: white;
  padding: 20px;
  border-radius: 5px;
  max-width: 80%;
  max-height: 80%;
  overflow-y: auto;
}

h2 {
  text-align: center;
  margin-bottom: 20px;
}

.rules-text {
  text-align: left;
}

ul {
  padding-left: 20px;
}

.close-button {
  display: block;
  margin: 20px auto 0;
  padding: 10px 20px;
  font-size: 16px;
  cursor: pointer;
}
</style>
