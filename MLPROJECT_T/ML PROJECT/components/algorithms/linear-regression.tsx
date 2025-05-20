"use client"

import React, { useEffect, useRef, useState } from "react";
import { BlockMath } from "react-katex"; // Import BlockMath from react-katex

type Params = {
  learningRate: number;
  iterations: number;
  noise: number;
};

export default function LinearRegressionPage() {
  const [params, setParams] = useState<Params>({
    learningRate: 0.1,
    iterations: 100,
    noise: 0.2,
  });

  const [isAnimating, setIsAnimating] = useState(false);
  const [frame, setFrame] = useState(0);
  const reqRef = useRef<number | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const paramControls = [
    { name: "learningRate", label: "Learning Rate", min: 0.01, max: 0.5, step: 0.01 },
    { name: "iterations", label: "Iterations", min: 10, max: 200, step: 10 },
    { name: "noise", label: "Noise", min: 0, max: 1, step: 0.05 },
  ] as const;

  type Control = (typeof paramControls)[number];

  useEffect(() => {
    const c = canvasRef.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    renderLinearRegression(ctx, c.width, c.height, params, frame);
  }, [params, frame]);

  useEffect(() => {
    if (!isAnimating || frame >= params.iterations) {
      if (reqRef.current !== null) cancelAnimationFrame(reqRef.current);
      return;
    }

    const loop = () => {
      setFrame((f) => f + 1);
      reqRef.current = requestAnimationFrame(loop);
    };

    reqRef.current = requestAnimationFrame(loop);
    return () => {
      if (reqRef.current !== null) cancelAnimationFrame(reqRef.current);
    };
  }, [isAnimating, frame, params.iterations]);

  const renderLinearRegression = (
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    params: Params,
    frame = 0
  ) => {
    const { learningRate, iterations, noise } = params;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, width, height);

    const margin = 40;
    const plotWidth = width - margin * 2;
    const plotHeight = height - margin * 2;

    const points: { x: number; y: number }[] = [];
    for (let i = 0; i < 100; i++) {
      const x = Math.random() * 2 - 1;
      const y = 2 * x + 1 + (Math.random() * 2 - 1) * noise;
      points.push({ x, y });
    }

    // Gradient descent
    let w = 0, b = 0;
    const currentIteration = Math.min(
      iterations,
      Math.floor((frame / 100) * iterations) + 1
    );

    for (let i = 0; i < currentIteration; i++) {
      let dw = 0, db = 0;
      for (const pt of points) {
        const yHat = w * pt.x + b;
        const error = yHat - pt.y;
        dw += pt.x * error;
        db += error;
      }
      dw /= points.length;
      db /= points.length;

      w -= learningRate * dw;
      b -= learningRate * db;
    }

    // Draw axes
    ctx.strokeStyle = "#666";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(margin, height / 2);
    ctx.lineTo(width - margin, height / 2);
    ctx.moveTo(width / 2, margin);
    ctx.lineTo(width / 2, height - margin);
    ctx.stroke();

    // Plot line
    ctx.strokeStyle = "#ff3860";
    ctx.lineWidth = 4;
    ctx.beginPath();
    for (let px = 0; px <= plotWidth; px += 1) {
      const x = (px / plotWidth) * 2 - 1;
      const y = w * x + b;
      const cx = margin + px;
      const cy = height / 2 - y * plotHeight / 2;
      px === 0 ? ctx.moveTo(cx, cy) : ctx.lineTo(cx, cy);
    }
    ctx.stroke();

    // Points
    for (const pt of points) {
      const cx = margin + ((pt.x + 1) / 2) * plotWidth;
      const cy = height / 2 - pt.y * plotHeight / 2;
      ctx.beginPath();
      ctx.arc(cx, cy, 4, 0, Math.PI * 2);
      ctx.fillStyle = "#34d399";
      ctx.fill();
    }

    // MSE
    const mse =
      points.reduce((sum, pt) => {
        const pred = w * pt.x + b;
        return sum + (pt.y - pred) ** 2;
      }, 0) / points.length;

    ctx.fillStyle = "#fff";
    ctx.font = "14px sans-serif";
    ctx.fillText(`Weight: ${w.toFixed(3)}`, margin + 10, margin + 20);
    ctx.fillText(`Bias: ${b.toFixed(3)}`, margin + 10, margin + 40);
    ctx.fillText(`Iterations: ${currentIteration}/${iterations}`, margin + 10, margin + 60);
    ctx.fillText(`MSE: ${mse.toFixed(3)}`, margin + 10, margin + 80);
  };

  return (
    <div className="space-y-8 p-4 bg-black text-white">
      <h1 className="text-4xl font-bold">Linear Regression</h1>
      <p className="text-gray-300">
        A simple model that predicts continuous values using a linear relationship.
      </p>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2">
          <canvas
            ref={canvasRef}
            width={1000} // Increased canvas width
            height={600} // Increased canvas height
            className="w-full border border-gray-700"
          />
          <button
            onClick={() => {
              setIsAnimating((a) => {
                if (!a) setFrame(0);
                return !a;
              });
            }}
            className="mt-2 px-4 py-2 bg-black text-white rounded border border-white hover:bg-white hover:text-black transition"
          >
            {isAnimating ? "Pause" : "Animate"}
          </button>

          <section className="controls-section">
            <h3 className="text-xl font-semibold text-white mb-4">Controls</h3>

            {paramControls.map((ctl) => {
              const val = params[ctl.name];
              const pct = ((val - ctl.min) / (ctl.max - ctl.min)) * 100;
              return (
                <div key={ctl.name} className="control-row">
                  <span className="control-label">{ctl.label}</span>
                  <input
                    type="range"
                    min={ctl.min}
                    max={ctl.max}
                    step={ctl.step}
                    value={val}
                    onChange={(e) =>
                      setParams((p) => ({
                        ...p,
                        [ctl.name]: Number(e.target.value),
                      }))
                    }
                    className="slider"
                    style={{ "--value": `${pct}%` } as React.CSSProperties}
                  />
                  <span className="control-value">{val}</span>
                </div>
              );
            })}
          </section>
        </div>

        <div className="space-y-6">
          <h2 className="text-2xl font-semibold">How It Works</h2>
          <p>
            Linear regression fits a straight line to data by minimizing the mean squared error between predictions and true values.
          </p>
          <h3 className="text-xl font-semibold">Formula</h3>
          <div className="text-2xl">
  <BlockMath math="y = wx + b" />
</div>

          <p>
            It learns <code>w</code> and <code>b</code> using gradient descent.
          </p>
          <h3 className="text-xl font-semibold">Applications</h3>
          <ul className="list-disc list-inside">
            <li>Forecasting</li>
            <li>Risk analysis</li>
            <li>Real estate pricing</li>
            <li>Trend prediction</li>
          </ul>
        </div>
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
  );
}
