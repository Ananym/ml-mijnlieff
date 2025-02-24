<template>
  <div id="app">
    <h1>ML Mijnlieff</h1>
    <template v-if="!gameStarted">
      <p class="intro-text">Play against an ML-powered opponent</p>
      <p class="designer-credit">
        Game designed by
        <a href="https://www.hopwoodgames.com/mijnlieff-game-page" target="_blank">Andy Hopwood</a>
      </p>
      <div v-if="apiError" class="api-error">{{ apiError }}</div>
      <button @click="showRules">Show Rules</button>
      <h2>Select Difficulty:</h2>
      <div class="difficulty-selector">
        <button
          v-for="diff in sortedDifficulties"
          :key="diff.level"
          @click="selectDifficulty(diff.level)"
          :class="{ selected: selectedDifficulty === diff.level }"
        >
          {{ diff.name }}
        </button>
      </div>
      <h2>Choose your player:</h2>
      <button @click="startGame(true)">Go First</button>
      <button @click="startGame(false)">Go Second</button>
    </template>
    <template v-else>
      <div class="game-container" v-if="gameState !== null">
        <GameBoard
          :board="gameState.board"
          :currentPlayer="gameState.currentPlayer"
          :validMoves="validMoves"
          :selectedPieceType="selectedPieceType"
          :lastAIMove="lastAIMove"
          :isAIThinking="isAIThinking"
          :pieceCounts="gameState.pieceCounts"
          :isHumanFirst="isHumanFirst"
          @make-move="onLocalHumanMakeMove"
        />
        <PieceSupplySelector
          :pieceCounts="gameState.pieceCounts"
          :isHumanFirst="isHumanFirst"
          :selectedPieceType="selectedPieceType"
          :disabled="gameOver"
          @select-piece="selectPiece"
        />
      </div>
      <div class="game-controls">
        <button @click="restartGame" :disabled="isAIThinking">Restart Game</button>
        <button @click="showRules" :disabled="isAIThinking">Rules</button>
        <div v-show="isAIThinking" :class="{ hidden: !isAIThinking }">AI is thinking...</div>
        <!-- <div>
          Current player is
          {{ gameState.currentPlayer === Player.ONE ? 'Player One' : 'Player Two' }}
        </div> -->
      </div>
      <div v-if="prediction !== null && !gameOver">AI victory prediction: {{ prediction }}</div>
      <div v-if="gameOver">
        <h2>Game Over! {{ winnerProclamation }}</h2>
        <p>Human Score: {{ scores.HUMAN }}</p>
        <p>AI Score: {{ scores.AI }}</p>
      </div>
    </template>
    <Rules :show="rulesVisible" @close="hideRules" />
  </div>
</template>

<script lang="ts">
import { ref, computed, Ref } from 'vue';
import GameBoard from './components/GameBoard.vue';
import PieceSupplySelector from './components/PieceSupplySelector.vue';
import Rules from './components/Rules.vue';
import {
  initializeGameState,
  makeMove,
  getValidMoves,
  calculateScores,
  TurnResults,
  TurnResult,
  PieceTypes,
  PieceType,
  Players,
  Move,
  GameState,
  Player,
  GameTypes,
  PieceNames,
  IllegalMoveException,
  getStateRepresentation,
  getDebugStringRepresentation,
  moveToString,
  passTurn,
  DIFFICULTY_SETTINGS,
} from './gameLogic';

export default {
  name: 'App',
  components: {
    GameBoard,
    PieceSupplySelector,
    Rules,
  },
  setup() {
    const gameStarted = ref(false);
    const gameState: Ref<GameState | null> = ref(null);
    const selectedPieceType: Ref<PieceType> = ref(PieceTypes.DIAG);
    const gameOver = ref(false);
    const scores: Ref<{ HUMAN: number; AI: number }> = ref({ HUMAN: 0, AI: 0 });
    const isAIThinking: Ref<boolean> = ref(false);
    const isHumanFirst: Ref<boolean> = ref(true);
    const lastAIMove: Ref<Move | null> = ref(null);
    const prediction: Ref<number | null> = ref(null);
    const rulesVisible: Ref<boolean> = ref(false);
    const selectedDifficulty: Ref<number> = ref(0); // Default to hardest
    const difficulties = DIFFICULTY_SETTINGS;
    const apiError: Ref<string | null> = ref(null);

    const useRandomDummy = false;

    const sortedDifficulties = computed(() => {
      return Object.entries(difficulties)
        .map(([level, settings]) => ({ level: Number(level), ...settings }))
        .sort((a, b) => b.level - a.level);
    });

    const showRules = () => {
      rulesVisible.value = true;
    };

    const hideRules = () => {
      rulesVisible.value = false;
    };

    const validMoves = computed(() => (gameState.value ? getValidMoves(gameState.value) : []));
    const winnerProclamation = computed(() => {
      if (!gameOver.value) return '';
      if (scores.value.HUMAN > scores.value.AI) return 'Human wins!';
      if (scores.value.AI > scores.value.HUMAN) return 'AI wins!';
      return "It's a tie!";
    });

    const pingApi = async () => {
      try {
        const response = await fetch(`${import.meta.env.VITE_API_ENDPOINT}/ping`, {
          method: 'GET',
        });
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        apiError.value = null;
      } catch (error) {
        apiError.value = 'AI service unavailable. Please try again later.';
        console.error('API ping failed:', error);
      }
    };

    const startGame = async (humanFirst: boolean) => {
      // Start ping check but don't wait for it
      pingApi();

      gameStarted.value = true;
      isHumanFirst.value = humanFirst;
      const aiPlayer = humanFirst ? Players.TWO : Players.ONE;
      gameState.value = initializeGameState(GameTypes.AI_VS_HUMAN);
      if (!humanFirst) {
        playAiTurn();
        // this is the first turn so passing and winning are impossible
      }
    };

    const selectPiece = (pieceType: PieceType) => {
      selectedPieceType.value = pieceType;
      // console.log(`Human selected piece type: ${PieceNames[pieceType]}`);
    };

    const cycleToNextAvailablePieceIfNecessary = (player: Player) => {
      if (gameState.value === null) {
        return;
      }
      const pieceCounts = gameState.value.pieceCounts[player];
      if (
        pieceCounts[selectedPieceType.value] !== 0 ||
        Object.values(pieceCounts).every((count) => count === 0)
      ) {
        // selected type has pieces remaining, do nothing
        // no pieces left, do nothing
        return;
      }

      for (let i = selectedPieceType.value + 1; i < selectedPieceType.value + 4; i++) {
        const pieceType: PieceType = (((i - 1) % 4) + 1) as PieceType;
        const count = pieceCounts[pieceType];
        if (count > 0) {
          selectedPieceType.value = pieceType;
          break;
        }
      }
    };

    const endGame = () => {
      if (gameState.value === null) {
        return;
      }
      gameOver.value = true;
      isAIThinking.value = false;
      const rawScores = calculateScores(gameState.value);

      const p1Score = rawScores[Players.ONE];
      const p2Score = rawScores[Players.TWO];
      const humanScore = isHumanFirst.value ? p1Score : p2Score;
      const aiScore = isHumanFirst.value ? p2Score : p1Score;

      scores.value.HUMAN = humanScore;
      scores.value.AI = aiScore;
    };

    const onLocalHumanMakeMove = async (x: number, y: number) => {
      console.log('Human has attempted a move');
      if (gameState.value === null) {
        return;
      }
      // Clear the prediction when human makes a move
      prediction.value = null;
      let turnResult;
      const move = { x, y, pieceType: selectedPieceType.value };
      try {
        turnResult = makeMove(gameState.value, move);
      } catch (e) {
        if (e instanceof IllegalMoveException) {
          // show a warning
          console.log('Invalid move: ' + e.message);
        } else {
          throw e;
        }
      }

      cycleToNextAvailablePieceIfNecessary(isHumanFirst.value ? Players.ONE : Players.TWO);

      console.log(moveToString(move, isHumanFirst.value ? Players.ONE : Players.TWO));
      console.log('Turn result: ' + turnResult);
      // console.log('Game state: ' + getDebugStringRepresentation(gameState.value));

      switch (turnResult) {
        case TurnResults.NORMAL:
          // if is vs ai
          console.log('Turn result is normal, so playing AI turn');
          await playAiTurn();
          break;
        case TurnResults.GAME_OVER:
          console.log('Turn result is game over, so ending game');
          lastAIMove.value = null;
          endGame();
          break;
        case TurnResults.OPP_MUST_PASS:
          lastAIMove.value = null;
          passTurn(gameState.value);
          break;
      }
    };

    const playAiTurn = async (): Promise<void> => {
      console.log('ai is attempting a move');
      if (gameState.value === null) {
        return;
      }
      // IN THEORY this is never called while game would be over
      // IN THEORY impossible to have no valid moves at start of turn

      while (true) {
        isAIThinking.value = true;
        let move: Move;
        // console.log(`Getting AI move...`);

        // console.log(JSON.stringify(gameState.value.pieceCounts));

        if (useRandomDummy) {
          const validMoves = getValidMoves(gameState.value);
          const randomIndex = Math.floor(Math.random() * validMoves.length);
          move = validMoves[randomIndex];
        } else {
          const response = await fetch(`${import.meta.env.VITE_API_ENDPOINT}/api/get_ai_move`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              ...getStateRepresentation(gameState.value),
              difficulty: selectedDifficulty.value,
            }),
          });

          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          const data = await response.json();
          prediction.value = Number(data.aiWinProbability.toPrecision(3));
          move = data;
        }

        const turnResult = makeMove(gameState.value, move);
        console.log(moveToString(move, isHumanFirst.value ? Players.TWO : Players.ONE));
        console.log('Turn result: ' + turnResult);
        console.log('Game state: ' + getDebugStringRepresentation(gameState.value));
        lastAIMove.value = move;
        isAIThinking.value = false;

        switch (turnResult) {
          case TurnResults.NORMAL:
            return;
          case TurnResults.GAME_OVER:
            endGame();
            return;
          case TurnResults.OPP_MUST_PASS:
            passTurn(gameState.value);
            continue;
        }
      }
    };

    const restartGame = () => {
      console.log('Game restarted.');
      gameStarted.value = false;
      gameState.value = initializeGameState(GameTypes.AI_VS_HUMAN);
      selectedPieceType.value = PieceTypes.DIAG;
      gameOver.value = false;
      scores.value = { HUMAN: 0, AI: 0 };
      isAIThinking.value = false;
      lastAIMove.value = null;
      prediction.value = null;
    };

    const selectDifficulty = (level: number) => {
      selectedDifficulty.value = level;
    };

    // Start ping check on mount but don't block
    pingApi();

    return {
      gameStarted,
      gameState,
      selectedPieceType,
      validMoves,
      gameOver,
      scores,
      prediction,
      isAIThinking,
      isHumanFirst,
      lastAIMove,
      startGame,
      selectPiece,
      onLocalHumanMakeMove,
      restartGame,
      rulesVisible,
      showRules,
      hideRules,
      winnerProclamation,
      Players,
      difficulties,
      selectedDifficulty,
      selectDifficulty,
      sortedDifficulties,
      apiError,
    };
  },
};
</script>

<style>
:root {
  --bg-color: #ffffff;
  --text-color: #333333;
  --link-color: #4caf50;
  --link-hover-color: #45a049;
  --button-bg: #f0f0f0;
  --button-border: #ddd;
  --button-hover-bg: #e0e0e0;
  --button-selected-bg: #4caf50;
  --button-selected-color: white;
  --button-selected-border: #45a049;
  --button-selected-hover: #45a049;
  --button-disabled-opacity: 0.6;
}

@media (prefers-color-scheme: dark) {
  :root {
    --bg-color: #242424;
    --text-color: #e0e0e0;
    --link-color: #6abf6e;
    --link-hover-color: #7ccf80;
    --button-bg: #2d2d2d;
    --button-border: #404040;
    --button-hover-bg: #3d3d3d;
    --button-selected-bg: #4caf50;
    --button-selected-color: #ffffff;
    --button-selected-border: #45a049;
    --button-selected-hover: #45a049;
    --button-disabled-opacity: 0.4;
  }
}

html,
body {
  margin: 0;
  padding: 0;
  background-color: var(--bg-color);
}

#app {
  font-family: Arial, sans-serif;
  text-align: center;
  padding: 0;
  color: var(--text-color);
  min-height: 100vh;
  background-color: transparent;
}

.intro-text {
  font-size: 1.2em;
  margin: 10px 0;
  color: var(--text-color);
}

.designer-credit {
  margin: 10px 0 20px;
  font-style: italic;
}

.designer-credit a {
  color: var(--link-color);
  text-decoration: none;
}

.designer-credit a:hover {
  color: var(--link-hover-color);
  text-decoration: underline;
}

.start-screen-controls {
  margin-top: 20px;
}

.game-container {
  display: flex;
  justify-content: center;
  align-items: center;
}

button {
  margin: 10px;
  padding: 10px 20px;
  font-size: 16px;
  cursor: pointer;
  background-color: var(--button-bg);
  border: 2px solid var(--button-border);
  color: var(--text-color);
  border-radius: 5px;
  transition: all 0.2s;
}

button:hover:not(:disabled) {
  background-color: var(--button-hover-bg);
}

button:disabled {
  cursor: not-allowed;
  opacity: var(--button-disabled-opacity);
}

.game-controls {
  display: flex;
  justify-content: center;
  align-items: center;
}

.difficulty-selector {
  display: flex;
  justify-content: center;
  gap: 10px;
  margin-bottom: 20px;
}

.difficulty-selector button {
  background-color: var(--button-bg);
  border: 2px solid var(--button-border);
  border-radius: 5px;
  transition: all 0.2s;
}

.difficulty-selector button.selected {
  background-color: var(--button-selected-bg);
  color: var(--button-selected-color);
  border-color: var(--button-selected-border);
}

.difficulty-selector button:hover {
  background-color: var(--button-hover-bg);
}

.difficulty-selector button.selected:hover {
  background-color: var(--button-selected-hover);
}

.api-error {
  color: #ff4444;
  margin: 10px 0;
  padding: 10px;
  background-color: rgba(255, 68, 68, 0.1);
  border-radius: 5px;
}
</style>
