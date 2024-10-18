<!-- <template>
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40">
    <g>
      <path
        d="M5 5 L35 35 M35 5 L5 35"
        stroke="currentColor"
        stroke-width="2"
        fill="none"
      />
    </g>
  </svg>
</template> -->

<!-- <template>
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40">
    <g>
      <path
        d="M5 5 L35 35 M35 5 L5 35"
        stroke="currentColor"
        stroke-width="2"
        fill="none"
      />
    </g>
  </svg>
</template> -->

<template>
  <svg
    class="rotated-icon"
    width="40"
    height="40"
    viewBox="0 0 40 40"
    xmlns="http://www.w3.org/2000/svg"
  >
    <g transform="rotate(45 20 20)">
      <OrthoIcon />
    </g>
  </svg>
</template>

<script lang="ts">
import OrthoIcon from "./OrthoIcon.vue";

export default {
  components: {
    OrthoIcon,
  },
};
</script>

<!-- <style scoped>
.rotated-icon {
  display: inline-block;
}
</style> -->
