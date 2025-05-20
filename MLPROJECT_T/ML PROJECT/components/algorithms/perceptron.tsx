"use client";

import { useEffect, useRef, useState, useMemo } from "react";
import { BlockMath } from "react-katex";
import "katex/dist/katex.min.css";

type Layer = { weights: number[][]; biases: number[] };

const activations = {
  sigmoid: {
    label: "Sigmoid",
    func: (x: number) => 1 / (1 + Math.exp(-x)),
    derivative: (y: number) => y * (1 - y),
    formula: "\\sigma(x) = \\frac{1}{1 + e^{-x}}",
  },
  relu: {
    label: "ReLU",
    func: (x: number) => Math.max(0, x),
    derivative: (y: number) => (y > 0 ? 1 : 0),
    formula: "\\text{ReLU}(x) = \\max(0, x)",
  },
  identity: {
    label: "Identity",
    func: (x: number) => x,
    derivative: (_: number) => 1,
    formula: "\\phi(x) = x",
  },
};

export default function PerceptronVisualizer() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>();
  const networkRef = useRef<Layer[]>([]);
  const [config, setConfig] = useState({
    hiddenLayers: 2,
    iterations: 10,
    activation: "sigmoid",
    learningRate: 0.1,
    outputNodes: 1,
  });

  const inputSize = 2,
    hiddenSize = 3;
  const input = [0.5, -0.3];
  const target = useMemo(() => Array(config.outputNodes).fill(1), [config.outputNodes]);

  const createNetwork = () => {
    const layers: Layer[] = [];
    let prev = inputSize;
    for (let i = 0; i < config.hiddenLayers; i++) {
      layers.push({
        weights: Array.from({ length: hiddenSize }, () =>
          Array.from({ length: prev }, () => (Math.random() * 2 - 1) * 0.5)
        ),
        biases: Array.from({ length: hiddenSize }, () => (Math.random() * 2 - 1) * 0.5),
      });
      prev = hiddenSize;
    }
    layers.push({
      weights: Array.from({ length: config.outputNodes }, () =>
        Array.from({ length: prev }, () => (Math.random() * 2 - 1) * 0.5)
      ),
      biases: Array.from({ length: config.outputNodes }, () => (Math.random() * 2 - 1) * 0.5),
    });
    return layers;
  };

  const forwardPass = (layers: Layer[]) => {
    const { func } = activations[config.activation as keyof typeof activations];
    const acts: number[][] = [input];
    let curr = input;
    for (const layer of layers) {
      const z = layer.weights.map((row, j) =>
        row.reduce((sum, w, i) => sum + w * curr[i], layer.biases[j])
      );
      curr = z.map(func);
      acts.push(curr);
    }
    return acts;
  };

  const draw = (layers: Layer[], acts: number[][]) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = (canvas.width = 800);
    const h = (canvas.height = 620);
    const r = 25;
    const marginX = 50;

    ctx.clearRect(0, 0, w, h);

    const layersCount = acts.length;
    const drawingWidth = w - marginX * 2;
    const xGap = drawingWidth / (layersCount - 1);

    const positions = acts.map((layer, li) => {
      const yGap = h / (layer.length + 1);
      return layer.map((_, ni) => ({
        x: marginX + li * xGap,
        y: (ni + 1) * yGap,
      }));
    });

    ctx.font = "11px monospace";
    ctx.textAlign = "center";
    for (let l = 0; l < layersCount - 1; l++) {
      const layer = layers[l];
      positions[l].forEach((p0, i) =>
        positions[l + 1].forEach((p1, j) => {
          ctx.strokeStyle = "rgba(255,255,255,0.3)";
          ctx.beginPath();
          ctx.moveTo(p0.x, p0.y);
          ctx.lineTo(p1.x, p1.y);
          ctx.stroke();

          const mx = (p0.x + p1.x) / 2;
          const my = (p0.y + p1.y) / 2;
          const dx = p1.x - p0.x,
            dy = p1.y - p0.y;
          const len = Math.hypot(dx, dy) || 1;
          const base = 20 + (j - i) * 10;
          const ox = (-dy / len) * base;
          const oy = (dx / len) * base;

          ctx.fillStyle = "white";
          ctx.fillText(layer.weights[j][i].toFixed(2), mx + ox, my + oy);
        })
      );

      const count = acts[l + 1].length;
      const mid = (count - 1) / 2;
      const yGap = h / (count + 1);
      positions[l + 1].forEach((p, j) => {
        const bx = p.x + (j - mid) * 50;
        const by = p.y - yGap / 2 - 15;
        ctx.strokeStyle = "orange";
        ctx.beginPath();
        ctx.moveTo(bx, by);
        ctx.lineTo(p.x, p.y);
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(bx, by, r * 0.6, 0, 2 * Math.PI);
        ctx.fillStyle = "orange";
        ctx.fill();
        ctx.strokeStyle = "white";
        ctx.stroke();

        ctx.fillStyle = "white";
        ctx.fillText(layers[l].biases[j].toFixed(2), bx, by + 4);
      });
    }

    positions.forEach((layerPos, li) =>
      layerPos.forEach((p, ni) => {
        const color =
          li === 0 ? "#22c55e" : li === layersCount - 1 ? "#f87171" : "#0ea5e9";
        ctx.beginPath();
        ctx.arc(p.x, p.y, r, 0, 2 * Math.PI);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = "white";
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.fillStyle = "white";
        ctx.fillText(acts[li][ni].toFixed(2), p.x, p.y + 6);
      })
    );

    const out = acts[acts.length - 1];
    const loss = out.reduce((sum, v, i) => sum + (v - target[i]) ** 2, 0) / out.length;
    ctx.fillStyle = "white";
    ctx.font = "17px sans-serif";
    ctx.textAlign = "end";
    ctx.fillText(`MSE: ${loss.toFixed(4)}`, w - 20, h - 20);
  };

  useEffect(() => {
    cancelAnimationFrame(animRef.current!);
    const newNetwork = createNetwork();
    networkRef.current = newNetwork;
  }, [config.hiddenLayers, config.outputNodes]);

  useEffect(() => {
    cancelAnimationFrame(animRef.current!)
  
    let layers = networkRef.current
    const lr = config.learningRate
    const { activation, iterations } = config
    const { derivative } = activations[activation as keyof typeof activations]
  
    let iter = 0
  
    const step = () => {
      const activationsList = forwardPass(layers)
      const deltas: number[][] = []
  
      // Output layer delta
      deltas.push(
        activationsList.at(-1)!.map((output, i) => (output - target[i]) * derivative(output))
      )
  
      // Hidden layer deltas (backpropagate)
      for (let l = layers.length - 2; l >= 0; l--) {
        const nextWeights = layers[l + 1].weights
        const nextDeltas = deltas[0]
        const currentActivations = activationsList[l + 1]
  
        deltas.unshift(
          currentActivations.map((a, j) =>
            nextWeights.reduce((sum, row, k) => sum + row[j] * nextDeltas[k], 0) * derivative(a)
          )
        )
      }
  
      // Update weights and biases
      layers = layers.map((layer, l) => ({
        weights: layer.weights.map((row, j) =>
          row.map((w, i) => w - lr * deltas[l][j] * activationsList[l][i])
        ),
        biases: layer.biases.map((b, j) => b - lr * deltas[l][j]),
      }))
  
      networkRef.current = layers
      draw(layers, activationsList)
  
      if (++iter < iterations) {
        animRef.current = requestAnimationFrame(step)
      }
    }
  
    step()
  
    return () => cancelAnimationFrame(animRef.current!)
  }, [
    config.activation,
    config.iterations,
    config.learningRate,
    config.hiddenLayers,
    config.outputNodes,
  ])
  


  return (
    <div className="flex p-6">
      <div className="flex-shrink-0">
        <h2 className="text-3xl font-bold text-white mb-4">MLP Visualizer</h2>
        <div className="p-4 bg-black border rounded w-[860px]">
          <canvas ref={canvasRef} className="w-full" />
        </div>

        <div className="mt-4 w-[880px] space-y-2">
        <div className="controls-section">
  {[
    {
      label: "Hidden Layers",
      key: "hiddenLayers",
      min: 1,
      max: 5,
      step: 1,
      value: config.hiddenLayers,
      percent: ((config.hiddenLayers - 1) / 4) * 100,
    },
    {
      label: "Iterations",
      key: "iterations",
      min: 1,
      max: 20,
      step: 1,
      value: config.iterations,
      percent: ((config.iterations - 1) / 19) * 100,
    },
    {
      label: "Learning Rate",
      key: "learningRate",
      min: 0.01,
      max: 1,
      step: 0.01,
      value: config.learningRate,
      percent: ((config.learningRate - 0.01) / 0.99) * 100,
    },
    {
      label: "Output Nodes",
      key: "outputNodes",
      min: 1,
      max: 5,
      step: 1,
      value: config.outputNodes,
      percent: ((config.outputNodes - 1) / 4) * 100,
    },
  ].map(({ label, key, min, max, step, value, percent }) => (
    <div className="control-row" key={key}>
      <label className="control-label">{label}</label>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        className="slider"
        onChange={(e) => setConfig((prev) => ({ ...prev, [key]: +e.target.value }))}
        style={{ "--value": `${percent}%` } as React.CSSProperties & { [key: string]: string }}
      />
      <span className="control-value">
        {key === "learningRate" ? value.toFixed(2) : value}
      </span>
    </div>
  ))}

  <div className="control-row">
    <label className="control-label">Activation</label>
    <select
      className="w-full bg-gray-700 text-gray-300 rounded p-1"
      value={config.activation}
      onChange={(e) => setConfig((prev) => ({ ...prev, activation: e.target.value }))}
    >
      {Object.entries(activations).map(([key, { label }]) => (
        <option key={key} value={key}>
          {label}
        </option>
      ))}
    </select>
  </div>
</div>

        </div>
      </div>

      <div className="ml-8 w-1/3 space-y-4">
        <h3 className="text-2xl font-semibold text-white">What is a Perceptron?</h3>
        <p className="text-gray-300">
          A perceptron is the simplest artificial neural network for binary classification,
          computing a weighted sum plus bias and applying an activation function.
        </p>
        <h3 className="text-xl font-semibold text-white">Formula</h3>
        <BlockMath math="y = \phi\left(\sum_i w_i x_i + b\right)" />
        <h3 className="text-xl font-semibold text-white">Activation Functions</h3>
        {Object.entries(activations).map(([key, { label, formula }]) => (
          <div key={key} className="text-gray-300">
            <strong>{label}:</strong>
            <BlockMath math={formula} />
          </div>
        ))}
        <h3 className="text-xl font-semibold text-white">Data Types</h3>
        <ul className="list-disc list-inside text-gray-300">
          <li>Binary inputs</li>
          <li>Continuous features</li>
          <li>Labels {"{-1,+1}"} or {"{0,1}"}</li>
        </ul>
        <h3 className="text-xl font-semibold text-white">Controls</h3>
        <p className="text-gray-300">
          <strong>Hidden Layers:</strong> Depth (1–5)<br />
          <strong>Iterations:</strong> Epochs<br />
          <strong>Learning Rate:</strong> Step size<br />
          <strong>Output Nodes:</strong> Number of outputs<br />
          <strong>Activation:</strong> Sigmoid, ReLU, or Identity
        </p>
        <h3 className="text-xl font-semibold text-white">Applications</h3>
        <ul className="list-disc list-inside text-gray-300">
          <li>Linearly separable classification</li>
          <li>Feature detection</li>
          <li>Base for deep networks</li>
        </ul>
      </div>
      <style jsx global>{`
  .controls-section {
    margin-top: 1.5rem;
    width: 100%;
    max-width: 800px;
    margin-left: auto;
    margin-right: auto;
  }
  .control-row {
    display: flex;
    align-items: center;
    margin-bottom: 1rem;
  }
  .control-label {
    width: 150px;
    color: white;
    font-weight: 500;
  }
  .control-value {
    width: 40px;
    text-align: right;
    color: white;
    margin-left: 0.5rem;
  }
  .slider {
    -webkit-appearance: none;
    appearance: none;
    width: 100%;
    height: 8px;
    border-radius: 4px;
    background: linear-gradient(
      to right,
      #ff3860 0%,
      #ff3860 var(--value),
      #333333 var(--value),
      #333333 100%
    );
    margin: 0 0.75rem;
  }
  .slider::-moz-range-track {
    background: transparent;
  }
  .slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: black;
    border: 2px solid #ff3860;
    margin-top: -5px;
    cursor: pointer;
  }
  .slider::-moz-range-thumb {
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: #ff3860;
    border: 2px solid white;
    cursor: pointer;
  }
`}</style>

    </div>
  )
}
