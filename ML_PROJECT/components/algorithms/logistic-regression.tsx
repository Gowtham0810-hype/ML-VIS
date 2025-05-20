"use client";

import React, { useEffect, useRef, useState } from "react";
import "katex/dist/katex.min.css";
import { BlockMath } from "react-katex";

// 1) Define a Params type
type Params = {
  learningRate: number;
  iterations: number;
  decisionBoundary: number;
  noise: number;
};

export default function LogisticRegressionPage() {
  // 2) Use that type in your useState
  const [params, setParams] = useState<Params>({
    learningRate: 0.1,
    iterations: 100,
    decisionBoundary: 0.5,
    noise: 0.2,
  });
  const [isAnimating, setIsAnimating] = useState(false);
  const [frame, setFrame] = useState(0);

  // 3) requestAnimationFrame ref
  const reqRef = useRef<number | null>(null);

  // 4) paramControls now uses keyof Params
  const paramControls: {
    name: keyof Params;
    label: string;
    min: number;
    max: number;
    step: number;
  }[] = [
    { name: "learningRate", label: "Learning Rate", min: 0.01, max: 0.5, step: 0.01 },
    { name: "iterations",      label: "Iterations",     min: 10,   max: 200, step: 10   },
    { name: "decisionBoundary",label: "Decision Boundary", min: 0.1, max: 0.9, step: 0.05 },
    { name: "noise",           label: "Data Noise",     min: 0,    max: 0.5, step: 0.05 },
  ];

  // 5) The main rendering function
  const renderLogisticRegression = (
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    params: Params,
    frame = 0
  ) => {
    const { learningRate, iterations, decisionBoundary, noise } = params;

    // Clear & background
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, width, height);

    const margin = 40;
    const plotWidth = width - 2 * margin;
    const plotHeight = height - 2 * margin;

    // Generate synthetic data
    const points: { x: number; y: number }[] = [];
    for (let i = 0; i < 100; i++) {
      const x = Math.random() * 2 - 1;
      const trueProb = 1 / (1 + Math.exp(-5 * x));
      const noisyProb = Math.min(1,
        Math.max(0, trueProb + (Math.random() * 2 - 1) * noise)
      );
      points.push({ x, y: noisyProb > 0.5 ? 1 : 0 });
    }

    // Train logistic model (gradient descent)
    let w = 0, b = 0;
    const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));
    const currentIteration = Math.min(
      iterations,
      Math.floor((frame / 100) * iterations) + 1
    );

    for (let i = 0; i < currentIteration; i++) {
      let dw = 0, db = 0;
      for (const pt of points) {
        const z = w * pt.x + b;
        const a = sigmoid(z);
        const dz = a - pt.y;
        dw += pt.x * dz;
        db += dz;
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
    ctx.moveTo(margin, height - margin);
    ctx.lineTo(width - margin, height - margin);
    ctx.moveTo(margin, margin);
    ctx.lineTo(margin, height - margin);
    ctx.stroke();

    // Axis labels
    ctx.fillStyle = "#999";
    ctx.font = "12px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("-1", margin, height - margin + 20);
    ctx.fillText("0", margin + plotWidth / 2, height - margin + 20);
    ctx.fillText("1", width - margin, height - margin + 20);
    ctx.textAlign = "right";
    ctx.fillText("0", margin - 10, height - margin);
    ctx.fillText("0.5", margin - 10, height - margin - plotHeight / 2);
    ctx.fillText("1", margin - 10, margin);

    // Decision boundary
    ctx.strokeStyle = "rgba(255,255,255,0.5)";
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(margin, height - margin - plotHeight * decisionBoundary);
    ctx.lineTo(width - margin, height - margin - plotHeight * decisionBoundary);
    ctx.stroke();
    ctx.setLineDash([]);

    // Logistic curve
    ctx.strokeStyle = "#ff3366";
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let px = 0; px <= plotWidth; px += 2) {
      const x = (px / plotWidth) * 2 - 1;
      const y = sigmoid(w * x + b);
      const cx = margin + px;
      const cy = height - margin - y * plotHeight;
      px === 0 ? ctx.moveTo(cx, cy) : ctx.lineTo(cx, cy);
    }
    ctx.stroke();

    // Draw data points
    for (const pt of points) {
      const cx = margin + ((pt.x + 1) / 2) * plotWidth;
      const cy = height - margin - pt.y * plotHeight;
      ctx.fillStyle = pt.y === 1 ? "#4ade80" : "#f87171";
      ctx.beginPath();
      ctx.arc(cx, cy, 5, 0, Math.PI * 2);
      ctx.fill();
    }

    // Compute error rate
    const preds = points.map((pt) =>
      sigmoid(w * pt.x + b) > decisionBoundary ? 1 : 0
    );
    const errors = preds.reduce<number>(
      (acc, p, i) => acc + (p !== points[i].y ? 1 : 0),
      0
    );
    const errorRate = errors / points.length;

    // Draw stats (no literal $)
    ctx.fillStyle = "#fff";
    ctx.font = "14px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(`Weight: ${w.toFixed(3)}`, margin + 10, margin + 20);
    ctx.fillText(`Bias: ${b.toFixed(3)}`, margin + 10, margin + 40);
    ctx.fillText(`Iterations: ${currentIteration}/${iterations}`, margin + 10, margin + 60);
    ctx.fillText(`Error Rate: ${(errorRate * 100).toFixed(1)}%`, margin + 10, margin + 80);
  };

  // Canvas ref
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Redraw when params or frame change
  useEffect(() => {
    const c = canvasRef.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    renderLogisticRegression(ctx, c.width, c.height, params, frame);
  }, [params, frame]);

  // Animation loop with stopping condition
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

  return (
    <div className="space-y-8 p-4 bg-black text-white">
      <h1 className="text-4xl font-bold">Logistic Regression</h1>
      <p className="text-gray-300">
        A statistical model that uses a logistic function to model a binary dependent variable.
      </p>
  
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2">
          <canvas
            ref={canvasRef}
            width={800}
            height={500}
            className="w-full border border-gray-700"
          />
  
          <button
            onClick={() => {
              setIsAnimating((a) => {
                if (!a) setFrame(0); // reset frame only when starting animation
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
              const percentage =
                ((params[ctl.name] - ctl.min) / (ctl.max - ctl.min)) * 100;
  
              return (
                <div className="control-row" key={ctl.name}>
                  <span className="control-label">{ctl.label}</span>
                  <input
                    type="range"
                    min={ctl.min}
                    max={ctl.max}
                    step={ctl.step}
                    value={params[ctl.name]}
                    onChange={(e) =>
                      setParams((p) => ({
                        ...p,
                        [ctl.name]: Number(e.target.value),
                      }))
                    }
                    style={
                      {
                        "--value": `${percentage}%`,
                      } as React.CSSProperties
                    }
                    className="slider"
                  />
                  <span className="control-value">{params[ctl.name]}</span>
                </div>
              );
            })}
          </section>
        </div>
  
        <div className="space-y-6">
          <div>
            <h2 className="text-2xl font-semibold">How It Works</h2>
            <p>
              Logistic regression predicts the probability of an observation belonging to a certain class via a sigmoid function.
            </p>
          </div>
  
          <div>
            <h3 className="text-xl font-semibold">The Sigmoid Function</h3>
            <p>The key to logistic regression is the sigmoid function:</p>
            <div className="my-4 text-center">
              <BlockMath math="\sigma(z) = \frac{1}{1 + e^{-z}}" />
            </div>
            <p>Where</p>
                <BlockMath math="z = w x + b" />
          </div>
  
          <div>
            <h3 className="text-xl font-semibold">Applications</h3>
            <p>Logistic regression is widely used in:</p>
            <ul className="list-disc list-inside">
              <li>Medical diagnosis</li>
              <li>Credit scoring</li>
              <li>Marketing (predicting customer behavior)</li>
              <li>Spam detection</li>
            </ul>
          </div>
        </div>
      </div>
  
      {/* Custom slider styles */}
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
