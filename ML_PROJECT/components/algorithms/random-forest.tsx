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

export default function MultipleDecisionTreePage() {
  const [irisDataset, setIrisDataset] = useState<Point[]>([]);
  const [params, setParams] = useState({
    maxDepth: 3,
    minSamplesSplit: 2,
    criterion: "gini",
    treeHeight: 600,
    numberOfTrees: 3,
    subsampleRatio: 0.8,
    featureSubsetRatio: 0.8
  });

  const [selectedIndex, setSelectedIndex] = useState<number>(0);
  const [forest, setForest] = useState<Node[]>([]);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [individualVotes, setIndividualVotes] = useState<number[]>([]);

  const canvasRefs = useRef<(HTMLCanvasElement | null)[]>([]);


  const labelMap: Record<string, number> = {
    setosa: 0,
    versicolor: 1,
    virginica: 2,
  };

  const reverseLabelMap = Object.fromEntries(
    Object.entries(labelMap).map(([key, val]) => [val, key])
  );

  const featureNames = ["Sepal Length", "Sepal Width"];

  useEffect(() => {
    fetch("/data/irisDataset.json")
      .then((res) => res.json())
      .then((data) => setIrisDataset(data))
      .catch((err) => console.error("Failed to load dataset:", err));
  }, []);

  useEffect(() => {
    if (irisDataset.length === 0) return;
    const trees: Node[] = [];
    for (let i = 0; i < params.numberOfTrees; i++) {
      const sample = subsampleData(irisDataset, params.subsampleRatio);
      trees.push(buildTree(sample, 0, params.maxDepth, params.minSamplesSplit, params.criterion));
    }
    setForest(trees);
  }, [irisDataset, params]);

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

  function predict(tree: Node, point: Point): number {
    let node = tree;
    while (node.left && node.right) {
      const value = node.feature === 0 ? point.sepalLength : point.sepalWidth;
      node = value <= node.threshold! ? node.left : node.right;
    }
    return node.value!;
  }

  function majorityVote(forest: Node[], point: Point): { final: number, votes: number[] } {
    const individual = forest.map(tree => predict(tree, point));
    const votes = [0, 0, 0];
    individual.forEach(v => votes[v]++);
    const final = votes.indexOf(Math.max(...votes));
    return { final, votes: individual };
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

  function subsampleData(data: Point[], ratio: number) {
    const sampleSize = Math.floor(data.length * ratio);
    const shuffled = [...data].sort(() => 0.5 - Math.random());
    return shuffled.slice(0, sampleSize);
  }

  function selectRandomFeatureSubset(ratio: number) {
    const numFeatures = Math.max(1, Math.floor(featureNames.length * ratio));
    const allFeatures = [0, 1];
    const shuffled = [...allFeatures].sort(() => 0.5 - Math.random());
    return shuffled.slice(0, numFeatures);
  }

  function findBestSplit(data: Point[], criterion: string, minSamplesSplit: number, availableFeatures: number[]) {
    if (data.length < minSamplesSplit) return null;

    const impurityFn = criterion.toLowerCase() === "gini" ? calculateGini : calculateEntropy;
    const labels = data.map((p) => labelMap[p.species]);
    const currentImpurity = impurityFn(labels);

    let bestGain = 0;
    let bestFeature: number | null = null;
    let bestThreshold: number | null = null;
    let bestImpurity = currentImpurity;

    const features = availableFeatures;

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
      // Calculate most common class
      const counts: Record<number, number> = {};
      for (const lbl of labels) {
        counts[lbl] = (counts[lbl] || 0) + 1;
      }
      let maxCount = 0;
      let maxLabel = 0;
      for (const [label, count] of Object.entries(counts)) {
        if (count > maxCount) {
          maxCount = count;
          maxLabel = Number(label);
        }
      }
      node.value = maxLabel;
      
      const impurityFn = criterion.toLowerCase() === "gini" ? calculateGini : calculateEntropy;
      node.impurity = impurityFn(labels);
      return node;
    }

    // Select random subset of features for this node (for randomness in trees)
    const availableFeatures = selectRandomFeatureSubset(params.featureSubsetRatio);
    
    const split = findBestSplit(data, criterion, minSamplesSplit, availableFeatures);
    if (!split) {
      const labels = data.map((p) => labelMap[p.species]);
      // Calculate most common class
      const counts: Record<number, number> = {};
      for (const lbl of labels) {
        counts[lbl] = (counts[lbl] || 0) + 1;
      }
      let maxCount = 0;
      let maxLabel = 0;
      for (const [label, count] of Object.entries(counts)) {
        if (count > maxCount) {
          maxCount = count;
          maxLabel = Number(label);
        }
      }
      node.value = maxLabel;
      
      const impurityFn = criterion.toLowerCase() === "gini" ? calculateGini : calculateEntropy;
      node.impurity = impurityFn(labels);
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

  function renderDecisionTree(ctx: CanvasRenderingContext2D, width: number, height: number, treeIndex: number) {
    ctx.clearRect(0, 0, width, height);
    
    // Create different bootstrapped samples for each tree
    const subsampledData = subsampleData(irisDataset, params.subsampleRatio);
    const tree = buildTree(subsampledData, 0, params.maxDepth, params.minSamplesSplit, params.criterion);

    ctx.font = "14px Arial";
    ctx.fillStyle = "#fff";
    ctx.textAlign = "center";

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
      ctx.font = "11px Arial";
      ctx.textAlign = "center";
      if (node.impurity !== null) {
        ctx.fillText(node.impurity.toFixed(2), x, y + 4);
      }

      if (node.feature !== null && node.threshold !== null) {
        ctx.fillStyle = "#fff";
        ctx.font = "11px Arial";
        ctx.fillText(featureNames[node.feature], x, y - 35);
        ctx.fillText(`≤ ${node.threshold.toFixed(2)}`, x, y - 22);
      }
    }

    drawNode(tree, width / 2, 50, width / 5, 80, 0);
  }

  useEffect(() => {
    if (irisDataset.length === 0) return;
    
    canvasRefs.current.forEach((canvas, index) => {
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      renderDecisionTree(ctx, canvas.width, canvas.height, index);
    });
  }, [params, irisDataset]);

  // Initialize canvas refs array when number of trees changes
  useEffect(() => {
    canvasRefs.current = Array(params.numberOfTrees).fill(null);
  }, [params.numberOfTrees]);

  return (
    <div className="p-4 space-y-10">
      <h1 className="text-4xl font-bold mb-4">Random Forest Visualizer</h1>
      <p className="text-lg text-muted-foreground">
        Visualizing multiple decision trees with different random subsamples of the Iris dataset.
        Each tree classifies Iris flowers by splitting data based on features like sepal length and width.
      </p>

      <div className="grid grid-cols-1 gap-6">
        <div className="space-y-4">
          <h2 className="text-2xl font-semibold">Random Forest</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {Array.from({ length: params.numberOfTrees }).map((_, index) => (
              <div key={index} className="border border-gray-300 rounded-lg bg-black p-2">
                <canvas
                  ref={el => canvasRefs.current[index] = el}
                  width={400}
                  height={400}
                  className="w-full"
                />
              </div>
            ))}
          </div>
        </div>
        
        {irisDataset.length > 0 && (
          <div className="mt-6 space-y-4">
            <h2 className="text-2xl font-semibold">Majority Vote Prediction</h2>
            <label className="text-white">
              Pick a sample index (0 to {irisDataset.length - 1}):
              <input
                type="number"
                min={0}
                max={irisDataset.length - 1}
                value={selectedIndex}
                onChange={(e) => setSelectedIndex(Number(e.target.value))}
                className="ml-2 p-1 rounded bg-gray-800 text-white border border-gray-600"
              />
            </label>
            <button
              onClick={() => {
                const point = irisDataset[selectedIndex];
                const { final, votes } = majorityVote(forest, point);
                setPrediction(final);
                setIndividualVotes(votes);
              }}
              className="ml-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
            >
              Predict
            </button>
            {prediction !== null && (
              <div className="text-white mt-4">
                <p>
                  <strong>True Label:</strong> {irisDataset[selectedIndex].species}
                </p>
                <p>
                  <strong>Individual Votes:</strong> {individualVotes.map(v => reverseLabelMap[v]).join(", ")}
                </p>
                <p>
                  <strong>Majority Vote:</strong> {reverseLabelMap[prediction]}
                </p>
              </div>
            )}
          </div>
        )}

        <section className="controls-section">
          <h3 className="text-xl font-semibold text-white mb-4">Controls</h3>

          <div className="control-row">
            <span className="control-label">Number of Trees</span>
            <input
              type="range"
              min="1"
              max="9"
              value={params.numberOfTrees}
              onChange={(e) =>
                setParams({ ...params, numberOfTrees: +e.target.value })
              }
              style={
                {
                  "--value": `${((params.numberOfTrees - 1) / 8) * 100}%`,
                } as React.CSSProperties
              }
              className="slider"
            />
            <span className="control-value">{params.numberOfTrees}</span>
          </div>

          <div className="control-row">
            <span className="control-label">Subsample Ratio</span>
            <input
              type="range"
              min="50"
              max="100"
              value={params.subsampleRatio * 100}
              onChange={(e) =>
                setParams({ ...params, subsampleRatio: +e.target.value / 100 })
              }
              style={
                {
                  "--value": `${((params.subsampleRatio * 100 - 50) / 50) * 100}%`,
                } as React.CSSProperties
              }
              className="slider"
            />
            <span className="control-value">{(params.subsampleRatio * 100).toFixed(0)}%</span>
          </div>

          <div className="control-row">
            <span className="control-label">Feature Subset</span>
            <input
              type="range"
              min="50"
              max="100"
              value={params.featureSubsetRatio * 100}
              onChange={(e) =>
                setParams({ ...params, featureSubsetRatio: +e.target.value / 100 })
              }
              style={
                {
                  "--value": `${((params.featureSubsetRatio * 100 - 50) / 50) * 100}%`,
                } as React.CSSProperties
              }
              className="slider"
            />
            <span className="control-value">{(params.featureSubsetRatio * 100).toFixed(0)}%</span>
          </div>

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
              <option value="gini">Gini</option>
              <option value="entropy">Entropy</option>
            </select>
            <span className="control-value">{params.criterion}</span>
          </div>

          <button
            className="reset-btn"
            onClick={() =>
              setParams({
                maxDepth: 3,
                minSamplesSplit: 2,
                criterion: "gini",
                treeHeight: 600,
                numberOfTrees: 3,
                subsampleRatio: 0.8,
                featureSubsetRatio: 0.8
              })
            }
          >
            <FiRefreshCw className="inline-block mr-2 h-5 w-5" />
            Reset
          </button>
        </section>

        <div className="space-y-6">
          <section>
            <h2 className="text-2xl font-semibold">How It Works</h2>
            <p className="text-muted-foreground">
              This visualizer shows multiple decision trees, each trained on a random subset of the Iris dataset.
              Each node shows the impurity score (Gini or Entropy), and the predicted class is indicated by color.
              If a split occurs, the feature used and the threshold value are shown above the node.
            </p>
            <p className="text-muted-foreground mt-2">
              The trees differ because they're trained on different random subsets of the data and may use
              different random feature subsets at each node, similar to how Random Forests work.
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
              <li><strong>Number of Trees:</strong> Controls how many different trees are displayed.</li>
              <li><strong>Subsample Ratio:</strong> Percentage of the dataset used for training each tree, creating diversity among trees.</li>
              <li><strong>Feature Subset:</strong> Percentage of features considered at each split, adding randomness to the tree building process.</li>
              <li><strong>Max Depth:</strong> Controls the maximum depth of each tree. Increasing the depth allows trees to make more splits.</li>
              <li><strong>Min Samples Split:</strong> Defines the minimum number of samples required to split an internal node.</li>
              <li><strong>Criterion:</strong> The function used to measure the quality of a split (Gini or Entropy).</li>
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

      {/* ——— Embedded CSS ——— */}
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
    </div>
  );
}