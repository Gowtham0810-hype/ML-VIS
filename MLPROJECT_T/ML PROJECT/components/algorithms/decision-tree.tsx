"use client";

import { useState, useEffect, useRef } from "react";
import "katex/dist/katex.min.css";
import { BlockMath } from "react-katex";
import { FiRefreshCw } from 'react-icons/fi';

type Point = {
  sepalLength: number;
  sepalWidth: number;
  species: string;
};

type Node = {
  feature: number | null;
  threshold: number | null;
  impurity: number | null;
  left: Node | null;
  right: Node | null;
  value: number | null;
};

export default function DecisionTreePage() {
  const [irisDataset, setIrisDataset] = useState<Point[]>([]);
  const [params, setParams] = useState({
    maxDepth: 3,
    minSamplesSplit: 2,
    criterion: "gini",
    treeHeight: 600,
  });

  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const labelMap: Record<string, number> = {
    setosa: 0,
    versicolor: 1,
    virginica: 2,
  };

  const featureNames = ["Sepal Length", "Sepal Width"];

  useEffect(() => {
    fetch("/data/irisDataset.json")
      .then((res) => res.json())
      .then((data) => setIrisDataset(data))
      .catch((err) => console.error("Failed to load dataset:", err));
  }, []);

  function calculateGini(labels: number[]) {
    const counts: Record<number, number> = {};
    for (const lbl of labels) {
      counts[lbl] = (counts[lbl] || 0) + 1;
    }
    let impurity = 1;
    for (const count of Object.values(counts)) {
      const p = count / labels.length;
      impurity -= p * p;
    }
    return impurity;
  }

  function calculateEntropy(labels: number[]) {
    const counts: Record<number, number> = {};
    for (const lbl of labels) {
      counts[lbl] = (counts[lbl] || 0) + 1;
    }
    let entropy = 0;
    for (const count of Object.values(counts)) {
      const p = count / labels.length;
      entropy -= p * Math.log2(p);
    }
    return entropy;
  }

  function findBestSplit(data: Point[], criterion: string, minSamplesSplit: number) {
    if (data.length < minSamplesSplit) return null;

    const impurityFn = criterion === "gini" ? calculateGini : calculateEntropy;
    const labels = data.map((p) => labelMap[p.species]);
    const currentImpurity = impurityFn(labels);

    let bestGain = 0;
    let bestFeature: number | null = null;
    let bestThreshold: number | null = null;
    let bestImpurity = currentImpurity;

    const features = [0, 1];

    for (const feature of features) {
      const values = [...new Set(data.map((p) => (feature === 0 ? p.sepalLength : p.sepalWidth)))].sort((a, b) => a - b);

      for (let i = 0; i < values.length - 1; i++) {
        const threshold = (values[i] + values[i + 1]) / 2;
        const left = data.filter((p) => (feature === 0 ? p.sepalLength : p.sepalWidth) <= threshold);
        const right = data.filter((p) => (feature === 0 ? p.sepalLength : p.sepalWidth) > threshold);
        if (left.length === 0 || right.length === 0) continue;

        const leftLabels = left.map((p) => labelMap[p.species]);
        const rightLabels = right.map((p) => labelMap[p.species]);

        const leftImp = impurityFn(leftLabels);
        const rightImp = impurityFn(rightLabels);

        const weightedImp = (left.length / data.length) * leftImp + (right.length / data.length) * rightImp;
        const gain = currentImpurity - weightedImp;

        if (gain > bestGain) {
          bestGain = gain;
          bestFeature = feature;
          bestThreshold = threshold;
          bestImpurity = currentImpurity;
        }
      }
    }

    if (bestGain === 0) return null;

    return {
      feature: bestFeature,
      threshold: bestThreshold,
      impurity: bestImpurity,
    };
  }

  function buildTree(data: Point[], depth: number, maxDepth: number, minSamplesSplit: number, criterion: string): Node {
    const node: Node = {
      feature: null,
      threshold: null,
      impurity: null,
      left: null,
      right: null,
      value: null,
    };

    if (depth >= maxDepth || data.length < minSamplesSplit) {
      const labels = data.map((p) => labelMap[p.species]);
      const sum = labels.reduce((a, b) => a + b, 0);
      node.value = Math.round(sum / labels.length);
      node.impurity = 0;
      return node;
    }

    const split = findBestSplit(data, criterion, minSamplesSplit);
    if (!split) {
      const labels = data.map((p) => labelMap[p.species]);
      const sum = labels.reduce((a, b) => a + b, 0);
      node.value = Math.round(sum / labels.length);
      node.impurity = 0;
      return node;
    }

    node.feature = split.feature;
    node.threshold = split.threshold;
    node.impurity = split.impurity;

    const left = data.filter((p) => (split.feature === 0 ? p.sepalLength : p.sepalWidth) <= split.threshold!);
    const right = data.filter((p) => (split.feature === 0 ? p.sepalLength : p.sepalWidth) > split.threshold!);

    node.left = buildTree(left, depth + 1, maxDepth, minSamplesSplit, criterion);
    node.right = buildTree(right, depth + 1, maxDepth, minSamplesSplit, criterion);

    return node;
  }

  function renderDecisionTree(ctx: CanvasRenderingContext2D, width: number, height: number) {
    ctx.clearRect(0, 0, width, height);

    const tree = buildTree(irisDataset, 0, params.maxDepth, params.minSamplesSplit, params.criterion);

    function drawNode(node: Node | null, x: number, y: number, dx: number, dy: number, depth: number) {
      if (!node) return;

      const radius = 20;

      if (node.left) {
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x - dx, y + dy);
        ctx.strokeStyle = "#ccc";
        ctx.stroke();
        drawNode(node.left, x - dx, y + dy, dx / 2, dy, depth + 1);
      }

      if (node.right) {
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + dx, y + dy);
        ctx.strokeStyle = "#ccc";
        ctx.stroke();
        drawNode(node.right, x + dx, y + dy, dx / 2, dy, depth + 1);
      }

      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      const isLeaf = !node.left && !node.right;

      if (isLeaf) {
        ctx.fillStyle = node.value === 0 ? "red" : node.value === 1 ? "blue" : "green";
      } else {
        ctx.fillStyle = "purple";
      }

      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.stroke();

      ctx.fillStyle = "#fff";
      ctx.font = "12px Arial";
      ctx.textAlign = "center";
      ctx.fillText(node.impurity!.toFixed(2), x, y + 4);

      if (node.feature !== null && node.threshold !== null) {
        ctx.fillStyle = "#fff";
        ctx.font = "12px Arial";
        ctx.fillText(featureNames[node.feature], x, y - 35);
        ctx.fillText(`≤ ${node.threshold.toFixed(2)}`, x, y - 22);
      }
    }

    drawNode(tree, width / 2, 50, width / 5, 80, 0);
  }

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    renderDecisionTree(ctx, canvas.width, canvas.height);
  }, [params, irisDataset]);

  return (
    <div className="p-4 space-y-10">
      <h1 className="text-4xl font-bold mb-4">Decision Tree Visualizer</h1>
      <p className="text-lg text-muted-foreground">
        A decision tree classifies Iris flowers by splitting data based on features like sepal length and width.
      </p>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <canvas
            ref={canvasRef}
            width={700}
            height={params.treeHeight - 100} 
            className="w-full border border-gray-300 rounded-lg bg-black"
          />

<section className="controls-section">
          <h3 className="text-xl font-semibold text-white mb-4">Controls</h3>

          <div className="control-row">
            <span className="control-label">Max Depth</span>
            <input
              type="range"
              min="1"
              max="6"
              value={params.maxDepth}
              onChange={(e) =>
                setParams({ ...params, maxDepth: +e.target.value })
              }
              style={
                {
                  "--value": `${((params.maxDepth - 1) / 5) * 100}%`,
                } as React.CSSProperties
              }
              className="slider"
            />
            <span className="control-value">{params.maxDepth}</span>
          </div>

          <div className="control-row">
            <span className="control-label">Min Samples Split</span>
            <input
              type="range"
              min="2"
              max="10"
              value={params.minSamplesSplit}
              onChange={(e) =>
                setParams({ ...params, minSamplesSplit: +e.target.value })
              }
              style={
                {
                  "--value": `${((params.minSamplesSplit - 2) / 8) * 100}%`,
                } as React.CSSProperties
              }
              className="slider"
            />
            <span className="control-value">{params.minSamplesSplit}</span>
          </div>

          <div className="control-row">
            <span className="control-label">Criterion</span>
            <select
              value={params.criterion}
              onChange={(e) =>
                setParams({ ...params, criterion: e.target.value })
              }
              className="styled-select"
            >
              <option value="Gini">Gini</option>
              <option value="Entropy">Entropy</option>
            </select>
            <span className="control-value">{params.criterion}</span>
          </div>

          <button
            className="reset-btn"
            onClick={() =>
              setParams({
                ...params,
                maxDepth: 3,
                minSamplesSplit: 2,
                criterion: "Gini",
              })
            }
          >
            <FiRefreshCw className="inline-block mr-2 h-5 w-5" />
            Reset
          </button>
        </section>
      </div>

      {/* ——— here’s your embedded CSS ——— */}
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
        .styled-select {
          flex: 1;
          margin: 0 0.75rem;
          padding: 6px 8px;
          background: #222;
          color: white;
          border: 1px solid #555;
          border-radius: 4px;
          appearance: none;
        }
        .styled-select::-ms-expand {
          display: none;
        }
        .reset-btn {
          background: black;
          color: white;
          padding: 0.5rem 1rem;
          border: 1px solid #444;
          border-radius: 4px;
          cursor: pointer;
        }
        .reset-btn:hover {
          border-color: #666;
        }
      `}</style>

        <div className="space-y-6">
          <section>
            <h2 className="text-2xl font-semibold">How It Works</h2>
            <p className="text-muted-foreground">
              Each node shows the impurity score (Gini or Entropy), and the predicted class is indicated by color.
              If a split occurs, the feature used is shown above the node.
            </p>
          </section>

          <section>
            <h3 className="text-xl font-semibold">Formulas</h3>
            <p className="text-muted-foreground">Gini Impurity:</p>
            <BlockMath math="Gini = 1 - \sum_{i=1}^{n} p_i^2" />
            <p className="text-muted-foreground">Information Gain:</p>
            <BlockMath math="Gain = Impurity(parent) - \sum_{k} \frac{n_k}{n} Impurity(k)" />
          </section>

          

          <section>
            <h3 className="text-xl font-semibold">Node Colors</h3>
            <ul className="list-disc list-inside text-muted-foreground">
              <li><span className="text-red-600 font-semibold">Red:</span> Setosa</li>
              <li><span className="text-blue-600 font-semibold">Blue:</span> Versicolor</li>
              <li><span className="text-green-600 font-semibold">Green:</span> Virginica</li>
              <li><span className="text-purple-600 font-semibold">Purple:</span> Internal (split) node</li>
            </ul>
          </section>

          <section>
            <h3 className="text-xl font-semibold">Interactive Controls</h3>
            <ul className="list-disc list-inside text-muted-foreground">
              <li><strong>Max Depth:</strong> Controls the maximum depth of the tree. Increasing the depth allows the tree to make more splits, but may lead to overfitting.</li>
              <li><strong>Min Samples Split:</strong> Defines the minimum number of samples required to split an internal node. A higher value can lead to simpler trees.</li>
              <li><strong>Criterion:</strong> The function used to measure the quality of a split. You can choose between "Gini" (Gini impurity) and "Entropy" (Information Gain).</li>
            </ul>
          </section>

          <section>
            <h3 className="text-xl font-semibold">Applications</h3>
            <p className="text-muted-foreground">Decision trees are widely used in:</p>
            <ul className="list-disc list-inside text-muted-foreground">
              <li>Banking (fraud detection, credit scoring)</li>
              <li>Medicine (disease diagnosis, treatment recommendation)</li>
              <li>E‑commerce (customer segmentation, recommendation systems)</li>
              <li>Computer Vision (object classification, image segmentation)</li>
            </ul>
          </section>

        </div>
      </div>
    </div>
  );
}
