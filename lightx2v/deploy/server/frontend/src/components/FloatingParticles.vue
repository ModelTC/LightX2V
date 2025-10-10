<template>
            <!-- 浮动粒子背景 -->
            <div class="floating-particles">
                <div class="particle"></div>
                <div class="particle"></div>
                <div class="particle"></div>
                <div class="particle"></div>
                <div class="particle"></div>
                <div class="particle"></div>
                <div class="particle"></div>
                <div class="particle"></div>
                <div class="particle"></div>
                <div class="particle"></div>
                <div class="particle"></div>
                <div class="particle"></div>
                <div class="particle"></div>
                <div class="particle"></div>
                <div class="particle"></div>
                <div class="particle"></div>
                <div class="particle"></div>
                <div class="particle"></div>
                <div class="particle"></div>
                <div class="particle"></div>
            </div>
</template>

<style scoped type="text/tailwindcss">
.floating-particles {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    overflow: hidden;
    pointer-events: none;
}

.particle {
    position: absolute;
    width: 4px;
    height: 4px;
    background: rgba(193, 169, 255, 0.6);
    border-radius: 50%;
    animation: floatParticle 15s linear infinite;
}

.particle:nth-child(1) { left: 10%; animation-delay: 0s; }
.particle:nth-child(2) { left: 20%; animation-delay: 12s; }
.particle:nth-child(3) { left: 30%; animation-delay: 10s; }
.particle:nth-child(4) { left: 40%; animation-delay: 6s; }
.particle:nth-child(5) { left: 50%; animation-delay: 8s; }
.particle:nth-child(6) { left: 60%; animation-delay: 14s; }
.particle:nth-child(7) { left: 70%; animation-delay: 16s; }
.particle:nth-child(8) { left: 80%; animation-delay: 2s; }
.particle:nth-child(9) { left: 90%; animation-delay: 4s; }

@keyframes floatParticle {
    0% {
        transform: translateY(100vh) scale(0);
        opacity: 0;
    }
    10% {
        opacity: 1;
    }
    90% {
        opacity: 1;
    }
    100% {
        transform: translateY(-100px) scale(1);
        opacity: 0;
    }
}
</style>
