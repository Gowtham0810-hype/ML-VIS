"use client";

import { useState, useEffect } from "react";
import VisualizationCanvas from "@/components/visualization-canvas";

export default function SOMPage() {
  const [params, setParams] = useState({
    gridSize: 10,
    learningRate: 0.1,
    iterations: 100,
    sigma: 1.0,
  });

  const [currentFrame, setCurrentFrame] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);

  const paramControls = [
    {
      name: "gridSize",
      label: "Grid Size",
      min: 5,
      max: 20,
      step: 1,
      defaultValue: 10,
    },
    {
      name: "learningRate",
      label: "Learning Rate",
      min: 0.01,
      max: 0.5,
      step: 0.01,
      defaultValue: 0.1,
    },
    {
      name: "iterations",
      label: "Iterations",
      min: 10,
      max: 200,
      step: 10,
      defaultValue: 100,
    },
    {
      name: "sigma",
      label: "Neighborhood Radius",
      min: 0.5,
      max: 3,
      step: 0.1,
      defaultValue: 1.0,
    },
  ];

  // Fixed dataset for visualization
  const fixedData = Array.from({ length: 200 }, () => {
    const angle = Math.random() * 2 * Math.PI;
    const radius = 0.6 + Math.random() * 0.2;
    return {
      x: Math.cos(angle) * radius,
      y: Math.sin(angle) * radius,
    };
  });

  const renderSOM = (
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    params: Record<string, any>,
    frame: number = 0
  ) => {
    const { gridSize, learningRate, iterations, sigma } = params;

    ctx.clearRect(0, 0, width, height);

    const margin = 40;
    const plotWidth = width - 2 * margin;
    const plotHeight = height - 2 * margin;

    // Coordinate conversion helpers
    const toCanvas = (x: number, y: number) => {
      return {
        x: margin + ((x + 1) / 2) * plotWidth,
        y: margin + ((y + 1) / 2) * plotHeight,
      };
    };

    // Init SOM grid
    const grid = [];
    for (let i = 0; i < gridSize; i++) {
      for (let j = 0; j < gridSize; j++) {
        grid.push({
          x: i / (gridSize - 1) - 0.5,
          y: j / (gridSize - 1) - 0.5,
          i,
          j,
        });
      }
    }

    const totalIterations = Math.min(iterations, frame);
    let lastBMU = null;

    // Train up to current frame
    for (let iter = 0; iter < totalIterations; iter++) {
      const point = fixedData[Math.floor(Math.random() * fixedData.length)];

      // Find BMU
      let bmuIndex = 0;
      let minDist = Infinity;
      for (let i = 0; i < grid.length; i++) {
        const node = grid[i];
        const dist = Math.hypot(point.x - node.x, point.y - node.y);
        if (dist < minDist) {
          minDist = dist;
          bmuIndex = i;
        }
      }

      lastBMU = grid[bmuIndex];

      const lrDecay = learningRate * Math.exp(-iter / iterations);
      const sigmaDecay = sigma * Math.exp(-iter / iterations);

      for (let i = 0; i < grid.length; i++) {
        const node = grid[i];
        const gridDist = Math.hypot(node.i - lastBMU.i, node.j - lastBMU.j);
        const influence = Math.exp(-(gridDist ** 2) / (2 * sigmaDecay ** 2));

        node.x += lrDecay * influence * (point.x - node.x);
        node.y += lrDecay * influence * (point.y - node.y);
      }
    }

    // Draw data points
    for (const point of fixedData) {
      const { x, y } = toCanvas(point.x, point.y);
      ctx.fillStyle = "rgba(96, 165, 250, 0.5)";
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, 2 * Math.PI);
      ctx.fill();
    }

    // Draw grid connections
    ctx.strokeStyle = "rgba(255,255,255,0.4)";
    ctx.lineWidth = 1;
    for (let i = 0; i < gridSize; i++) {
      for (let j = 0; j < gridSize; j++) {
        const idx = i * gridSize + j;
        const node = grid[idx];
        const nodeCanvas = toCanvas(node.x, node.y);

        if (j < gridSize - 1) {
          const right = grid[i * gridSize + j + 1];
          const rightCanvas = toCanvas(right.x, right.y);
          ctx.beginPath();
          ctx.moveTo(nodeCanvas.x, nodeCanvas.y);
          ctx.lineTo(rightCanvas.x, rightCanvas.y);
          ctx.stroke();
        }

        if (i < gridSize - 1) {
          const down = grid[(i + 1) * gridSize + j];
          const downCanvas = toCanvas(down.x, down.y);
          ctx.beginPath();
          ctx.moveTo(nodeCanvas.x, nodeCanvas.y);
          ctx.lineTo(downCanvas.x, downCanvas.y);
          ctx.stroke();
        }
      }
    }

    // Draw neurons
    for (const node of grid) {
      const { x, y } = toCanvas(node.x, node.y);
      ctx.fillStyle = "#ff3366";
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, 2 * Math.PI);
      ctx.fill();
    }

    // Highlight BMU with green circle
    if (lastBMU) {
      const { x, y } = toCanvas(lastBMU.x, lastBMU.y);
      ctx.strokeStyle = "#22c55e";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, 2 * Math.PI);
      ctx.stroke();
    }

    // Labels
    ctx.fillStyle = "#fff";
    ctx.font = "14px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(`Grid Size: ${gridSize}x${gridSize}`, margin, margin - 10);
    ctx.fillText(`Iterations: ${totalIterations}/${iterations}`, margin, margin + 10);
    ctx.fillText(`Learning Rate: ${learningRate}`, margin, margin + 30);
    ctx.fillText(`Radius (σ): ${sigma}`, margin, margin + 50);
  };

  // Use effect to animate the process
  useEffect(() => {
    let animationFrameId: number;
    let lastFrameTime = 0;
    const frameInterval = 16; // approximately 60fps

    const animate = (timestamp: number) => {
      // Control frame rate for smooth animation
      const elapsed = timestamp - lastFrameTime;
      if (elapsed > frameInterval) {
        lastFrameTime = timestamp;
        
        setCurrentFrame(prev => {
          const next = prev + 1;
          if (next <= params.iterations) {
            return next;
          } else {
            setIsAnimating(false); // Stop animation when complete
            return params.iterations;
          }
        });
      }

      if (isAnimating && currentFrame < params.iterations) {
        animationFrameId = requestAnimationFrame(animate);
      }
    };

    if (isAnimating && currentFrame < params.iterations) {
      animationFrameId = requestAnimationFrame(animate);
    }

    return () => {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
    };
  }, [isAnimating, currentFrame, params.iterations]);

  // Handle animation toggle with existing button
  const handleAnimateToggle = () => {
    if (currentFrame >= params.iterations) {
      // If animation completed, restart from beginning
      setCurrentFrame(0);
      setIsAnimating(true);
    } else {
      // Otherwise toggle animation state
      setIsAnimating(prev => !prev);
    }
  };

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-4xl font-bold mb-4">Self-Organizing Map</h1>
        <p className="text-lg text-muted-foreground mb-6">
          A type of artificial neural network that is trained using unsupervised learning to produce a low-dimensional
          representation of the input space.
        </p>
      </div>

      <div className="flex flex-col lg:flex-row gap-8">
        {/* Canvas Section (Left - 60%) */}
        <div className="lg:w-3/5">
          <VisualizationCanvas
            renderFunction={renderSOM}
            params={params}
            setParams={setParams}
            paramControls={paramControls}
            animate={true}
            height={500}
            currentFrame={currentFrame}
            onAnimateToggle={handleAnimateToggle}
          />
          
          {/* Progress indicator */}
          <div className="mt-4">
            <div className="w-full bg-gray-700 rounded-full h-2.5">
              <div 
                className="bg-blue-600 h-2.5 rounded-full" 
                style={{ width: `${(currentFrame / params.iterations) * 100}%` }}
              ></div>
            </div>
            <p className="text-sm text-gray-400 mt-1">
              Progress: {currentFrame}/{params.iterations} iterations
            </p>
          </div>
        </div>

        {/* Text Section (Right - 40%) */}
        <div className="space-y-6 lg:w-2/5">
          <div>
            <h2 className="text-2xl font-semibold mb-2">How It Works</h2>
            <p className="text-muted-foreground">
              Self-Organizing Maps (SOMs) are a type of artificial neural network that transform complex,
              high-dimensional data into a simpler low-dimensional representation while preserving the topological
              structure of the data.
            </p>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-2">Key Concepts</h3>
            <ul className="list-disc list-inside space-y-2 text-muted-foreground">
              <li>
                <strong>Competitive Learning:</strong> Neurons compete to be activated, with only one winning neuron
                being activated for each input.
              </li>
              <li>
                <strong>Best Matching Unit (BMU):</strong> The neuron whose weights are most similar to the input
                vector.
              </li>
              <li>
                <strong>Neighborhood Function:</strong> Determines how much neurons around the BMU are adjusted.
              </li>
              <li>
                <strong>Topology Preservation:</strong> Similar inputs activate neurons that are close to each other in
                the grid.
              </li>
            </ul>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-2">Algorithm Steps</h3>
            <ol className="list-decimal list-inside space-y-2 text-muted-foreground">
              <li>Initialize a grid of neurons with random weights.</li>
              <li>For each input vector, find the BMU (neuron with weights most similar to the input).</li>
              <li>Update the BMU and its neighbors to make them more similar to the input vector.</li>
              <li>Reduce the learning rate and neighborhood radius over time.</li>
              <li>Repeat until convergence or a maximum number of iterations.</li>
            </ol>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-2">Color Guide</h3>
            <ul className="list-disc list-inside space-y-2 text-muted-foreground">
              <li><span className="text-blue-400">Blue:</span> Input data points</li>
              <li><span className="text-pink-400">Pink:</span> SOM neurons</li>
              <li><span className="text-green-400">Green Circle:</span> Last Best Matching Unit (BMU)</li>
              <li><span className="text-white">White Lines:</span> Grid connections between neurons</li>
            </ul>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-2">Interactive Controls</h3>
            <ul className="list-disc list-inside space-y-2 text-muted-foreground">
              <li>
                <strong>Grid Size:</strong> The dimensions of the SOM grid. Larger grids can capture more detail but
                require more training.
              </li>
              <li>
                <strong>Learning Rate:</strong> Controls how much neurons are adjusted during training. Higher values
                lead to faster but potentially less stable learning.
              </li>
              <li>
                <strong>Iterations:</strong> The number of training steps. More iterations usually lead to better
                results but take longer.
              </li>
              <li>
                <strong>Neighborhood Radius:</strong> Controls how far the neighborhood extends around the BMU. Larger
                values affect more neurons.
              </li>
            </ul>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-2">Applications</h3>
            <p className="text-muted-foreground">SOMs are widely used in:</p>
            <ul className="list-disc list-inside space-y-2 text-muted-foreground">
              <li>Data visualization and dimensionality reduction</li>
              <li>Pattern recognition</li>
              <li>Image and speech processing</li>
              <li>Bioinformatics (gene expression analysis)</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}