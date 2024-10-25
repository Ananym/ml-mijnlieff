<template>
  <div id="app">
    <h1>Mijnlieff</h1>
    <template v-if="!gameStarted">
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
      <div v-if="!gameOver">
        Prediction: {{ prediction }}
      </div>
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

    const useRandomDummy = false;

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

    // const logRemainingPieces = (player) => {
    //   const pieces = gameState.value.pieceCounts[player];
    //   console.log(
    //     `Player ${player === Player.ONE ? "Human" : "AI"} pieces remaining:`,
    //     Object.entries(pieces)
    //       .map(([type, count]) => `${PieceNames[type]}: ${count}`)
    //       .join(", ")
    //   );
    // };

    // const logValidMoves = () => {
    //   console.log(
    //     `Valid moves for ${
    //       gameState.value.currentPlayer === Player.ONE ? "Human" : "AI"
    //     }:`,
    //     validMoves.value.length
    //   );
    // };

    const startGame = (humanFirst: boolean) => {
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
          const response = await fetch('http://localhost:5000/api/get_ai_move', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify(getStateRepresentation(gameState.value)),
          });

          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          const data = await response.json();
          prediction.value = data.prediction;
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
    };

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
    };
  },
};
</script>

<style scoped>
#app {
  font-family: Arial, sans-serif;
  text-align: center;
  padding: 0px;
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
}

button:disabled {
  cursor: not-allowed;
  opacity: 0.6;
}

.game-controls {
  display: flex;
  justify-content: center;
  align-items: center;
}
</style>
