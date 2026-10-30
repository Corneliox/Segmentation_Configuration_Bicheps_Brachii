/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#0b1120',
        panel: '#0f172a',
        card: '#1e293b',
        border: '#334155',
        accentCyan: '#00f2fe',
        accentYellow: '#fde047',
        accentRed: '#ff4444',
        accentOrange: '#f97316',
      },
    },
  },
  plugins: [],
}
