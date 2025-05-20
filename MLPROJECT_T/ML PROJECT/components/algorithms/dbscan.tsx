"use client"

import { useState, useMemo } from "react"
import VisualizationCanvas from "@/components/visualization-canvas"

const TOTAL_POINTS = 200
const NOISE_RATIO = 0.1

const generateClusterData = (numClusters: number, clusterPointCount: number) => {
  const clusterCenters = []
  const clusterPoints = []

  for (let i = 0; i < numClusters; i++) {
    const centerX = Math.random() * 1.6 - 0.8
    const centerY = Math.random() * 1.6 - 0.8
    clusterCenters.push({ x: centerX, y: centerY })
  }

  const pointsPerCluster = Math.floor(clusterPointCount / numClusters)

  for (let i = 0; i < numClusters; i++) {
    const center = clusterCenters[i]
    for (let j = 0; j < pointsPerCluster; j++) {
      const angle = Math.random() * Math.PI * 2
      const distance = Math.random() * 0.3

      const x = Math.max(-1, Math.min(1, center.x + Math.cos(angle) * distance))
      const y = Math.max(-1, Math.min(1, center.y + Math.sin(angle) * distance))

      clusterPoints.push({ x, y, cluster: -2 }) // -2 = unvisited
    }
  }

  return { clusterPoints }
}

export default function DBSCANPage() {
  const [params, setParams] = useState({
    epsilon: 0.2,
    minPoints: 5,
    numClusters: 3,
  })

  const { clusterPoints } = useMemo(() => {
    const clusterPointCount = Math.floor(TOTAL_POINTS * (1 - NOISE_RATIO))
    return generateClusterData(params.numClusters, clusterPointCount)
  }, [params.numClusters])

  const paramControls = [
    {
      name: "epsilon",
      label: "Epsilon (Neighborhood Radius)",
      min: 0.05,
      max: 0.5,
      step: 0.05,
      defaultValue: 0.2,
    },
    {
      name: "minPoints",
      label: "Min Points",
      min: 2,
      max: 10,
      step: 1,
      defaultValue: 5,
    },
    {
      name: "numClusters",
      label: "Number of Clusters",
      min: 1,
      max: 5,
      step: 1,
      defaultValue: 3,
    },
  ]

  const renderDBSCAN = (ctx: CanvasRenderingContext2D, width: number, height: number, params: Record<string, any>) => {
    const { epsilon, minPoints } = params

    const margin = 40
    const plotWidth = width - 2 * margin
    const plotHeight = height - 2 * margin

    ctx.clearRect(0, 0, width, height)

    // Copy static points
    const points = clusterPoints.map((p) => ({ ...p }))

    // Add noise (regenerated each render)
    const noiseCount = Math.floor(TOTAL_POINTS * NOISE_RATIO)
    for (let i = 0; i < noiseCount; i++) {
      const x = Math.random() * 2 - 1
      const y = Math.random() * 2 - 1
      points.push({ x, y, cluster: -1 }) // -1 = noise
    }

    // DBSCAN implementation (same as before)
    const dbscan = (points: any[], eps: number, minPts: number) => {
      let clusterIndex = 0

      const getNeighbors = (pointIndex: number) => {
        const neighbors = []
        const point = points[pointIndex]
        for (let i = 0; i < points.length; i++) {
          if (i === pointIndex) continue
          const otherPoint = points[i]
          const distance = Math.sqrt(Math.pow(point.x - otherPoint.x, 2) + Math.pow(point.y - otherPoint.y, 2))
          if (distance <= eps) neighbors.push(i)
        }
        return neighbors
      }

      const expandCluster = (pointIndex: number, neighbors: number[], cluster: number) => {
        points[pointIndex].cluster = cluster
        for (let i = 0; i < neighbors.length; i++) {
          const neighborIndex = neighbors[i]
          if (points[neighborIndex].cluster === -1) {
            points[neighborIndex].cluster = cluster
            const neighborNeighbors = getNeighbors(neighborIndex)
            if (neighborNeighbors.length >= minPts) {
              for (const nn of neighborNeighbors) {
                if (!neighbors.includes(nn)) neighbors.push(nn)
              }
            }
          } else if (points[neighborIndex].cluster === -2) {
            points[neighborIndex].cluster = cluster
          }
        }
      }

      for (const point of points) point.cluster = -2 // unvisited

      for (let i = 0; i < points.length; i++) {
        if (points[i].cluster !== -2) continue
        const neighbors = getNeighbors(i)
        if (neighbors.length < minPts) {
          points[i].cluster = -1
          continue
        }
        clusterIndex++
        expandCluster(i, neighbors, clusterIndex)
      }

      return clusterIndex
    }

    const numClustersFound = dbscan(points, epsilon, minPoints)

    // Draw axes
    ctx.strokeStyle = "#666"
    ctx.lineWidth = 1

    ctx.beginPath()
    ctx.moveTo(margin, height - margin)
    ctx.lineTo(width - margin, height - margin)
    ctx.stroke()

    ctx.beginPath()
    ctx.moveTo(margin, margin)
    ctx.lineTo(margin, height - margin)
    ctx.stroke()

    ctx.fillStyle = "#999"
    ctx.font = "12px sans-serif"
    ctx.textAlign = "center"
    ctx.fillText("-1", margin, height - margin + 20)
    ctx.fillText("0", margin + plotWidth / 2, height - margin + 20)
    ctx.fillText("1", width - margin, height - margin + 20)
    ctx.textAlign = "right"
    ctx.fillText("-1", margin - 10, height - margin)
    ctx.fillText("0", margin - 10, height - margin - plotHeight / 2)
    ctx.fillText("1", margin - 10, margin)

    const clusterColors = ["#4ade80", "#60a5fa", "#f472b6", "#fb923c", "#a78bfa"]

    for (const point of points) {
      const canvasX = margin + ((point.x + 1) / 2) * plotWidth
      const canvasY = margin + ((point.y + 1) / 2) * plotHeight
      let color
      if (point.cluster === -1) color = "#f87171"
      else color = clusterColors[(point.cluster - 1) % clusterColors.length]

      ctx.fillStyle = color
      ctx.beginPath()
      ctx.arc(canvasX, canvasY, 4, 0, Math.PI * 2)
      ctx.fill()

      if (Math.random() < 0.05) {
        ctx.strokeStyle = "rgba(255, 255, 255, 0.3)"
        ctx.beginPath()
        ctx.arc(canvasX, canvasY, (epsilon * plotWidth) / 2, 0, Math.PI * 2)
        ctx.stroke()
      }
    }

    ctx.fillStyle = "#fff"
    ctx.font = "14px sans-serif"
    ctx.textAlign = "left"
    ctx.fillText(`Epsilon: ${epsilon}`, margin + 10, margin + 20)
    ctx.fillText(`Min Points: ${minPoints}`, margin + 10, margin + 40)
    ctx.fillText(`Clusters Found: ${numClustersFound}`, margin + 10, margin + 60)
    ctx.fillText(`Noise Points: ${points.filter((p) => p.cluster === -1).length}`, margin + 10, margin + 80)
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-4xl font-bold mb-4">DBSCAN</h1>
        <p className="text-lg text-muted-foreground mb-6">
          Density-Based Spatial Clustering of Applications with Noise - a density-based clustering algorithm that groups
          together points that are closely packed together.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2">
          <VisualizationCanvas
            renderFunction={renderDBSCAN}
            params={params}
            setParams={setParams}
            paramControls={paramControls}
            height={500}
          />
        </div>

        <div className="space-y-6">
          <div>
            <h2 className="text-2xl font-semibold mb-2">How It Works</h2>
            <p className="text-muted-foreground">
              DBSCAN groups together points that are close to each other based on a distance measure (usually Euclidean
              distance) and a minimum number of points. It also marks as outliers (noise) points that are in low-density
              regions.
            </p>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-2">Key Concepts</h3>
            <ul className="list-disc list-inside space-y-2 text-muted-foreground">
              <li>
                <strong>Core Point:</strong> A point with at least minPoints points within distance epsilon.
              </li>
              <li>
                <strong>Border Point:</strong> A point within distance epsilon of a core point but with fewer than
                minPoints neighbors.
              </li>
              <li>
                <strong>Noise Point:</strong> A point that is neither a core point nor a border point.
              </li>
            </ul>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-2">Algorithm Steps</h3>
            <ol className="list-decimal list-inside space-y-2 text-muted-foreground">
              <li>Find all core points and their neighborhoods.</li>
              <li>Connect core points that are within epsilon distance of each other to form clusters.</li>
              <li>Assign each border point to the cluster of its closest core point.</li>
              <li>Label any remaining points as noise.</li>
            </ol>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-2">Interactive Controls</h3>
            <ul className="list-disc list-inside space-y-2 text-muted-foreground">
              <li>
                <strong>Epsilon:</strong> The maximum distance between two points for them to be considered neighbors.
                Larger values create larger clusters.
              </li>
              <li>
                <strong>Min Points:</strong> The minimum number of points required to form a dense region. Higher values
                require denser clusters.
              </li>
              <li>
                <strong>Number of Clusters:</strong> Controls how many clusters are generated in the synthetic data.
              </li>
            </ul>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-2">Applications</h3>
            <p className="text-muted-foreground">DBSCAN is widely used in:</p>
            <ul className="list-disc list-inside space-y-2 text-muted-foreground">
              <li>Anomaly detection</li>
              <li>Spatial data analysis</li>
              <li>Image segmentation</li>
              <li>Market research (customer segmentation)</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}
