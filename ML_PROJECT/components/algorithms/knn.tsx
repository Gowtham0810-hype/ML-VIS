"use client"

import React, { useEffect, useRef, useState } from "react"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import "katex/dist/katex.min.css"
import { InlineMath } from "react-katex"

const colors = ["red", "green", "blue", "orange", "purple", "cyan"]

export default function KNNVisualizer() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [config, setConfig] = useState({
    points: 100,
    classes: 3,
    k: 3,
  })
  const [data, setData] = useState<{ x: number; y: number; label: number }[]>([])
  const [testPoint, setTestPoint] = useState<{ x: number; y: number } | null>(null)
  const [prediction, setPrediction] = useState<number | null>(null)
  const [neighbors, setNeighbors] = useState<typeof data>([])

  useEffect(() => {
    const newData = Array.from({ length: config.points }, () => ({
      x: Math.random() * 780 + 10,
      y: Math.random() * 580 + 10,
      label: Math.floor(Math.random() * config.classes),
    }))
    setData(newData)
    setTestPoint(null)
    setPrediction(null)
    setNeighbors([])
  }, [config.points, config.classes])

  useEffect(() => {
    draw()
  }, [data, testPoint, prediction, neighbors])

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = canvasRef.current?.getBoundingClientRect()
    if (!rect) return
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    setTestPoint({ x, y })

    // Perform KNN
    const distances = data.map((d) => ({
      ...d,
      dist: (d.x - x) ** 2 + (d.y - y) ** 2,
    }))
    distances.sort((a, b) => a.dist - b.dist)
    const topK = distances.slice(0, config.k)
    setNeighbors(topK)

    const counts: Record<number, number> = {}
    for (const d of topK) {
      counts[d.label] = (counts[d.label] || 0) + 1
    }
    const predicted = Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0]
    setPrediction(parseInt(predicted))
  }

  const draw = () => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext("2d")
    if (!canvas || !ctx) return

    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Draw data points
    for (const p of data) {
      ctx.beginPath()
      ctx.arc(p.x, p.y, 5, 0, 2 * Math.PI)
      ctx.fillStyle = colors[p.label]
      ctx.fill()
    }

    // Highlight lines to neighbors
    if (testPoint && neighbors.length > 0) {
      ctx.strokeStyle = "#aaa"
      ctx.lineWidth = 1.5
      for (const n of neighbors) {
        ctx.beginPath()
        ctx.moveTo(testPoint.x, testPoint.y)
        ctx.lineTo(n.x, n.y)
        ctx.stroke()
      }
    }

    // Draw test point
    if (testPoint) {
      ctx.beginPath()
      ctx.arc(testPoint.x, testPoint.y, 8, 0, 2 * Math.PI)
      ctx.fillStyle = prediction !== null ? colors[prediction] : "gray"
      ctx.strokeStyle = "black"
      ctx.lineWidth = 2
      ctx.fill()
      ctx.stroke()
    }
  }

  const handleSliderChange = (key: keyof typeof config, value: number) => {
    setConfig((c) => ({ ...c, [key]: value }))
  }

  return (
    <div className="flex flex-col gap-6">
      <h1 className="text-4xl font-bold">KNN Visualizer</h1>
      <p className="text-medium text-muted-foreground">
        Click anywhere on the canvas to classify a new point using the KNN algorithm.
      </p>
      <div className="flex flex-col md:flex-row gap-8">
        <canvas
          ref={canvasRef}
          width={800}
          height={600}
          className="rounded-lg border bg-black"
          onClick={handleCanvasClick}
        />
        <div className="text-sm space-y-4 w-full md:w-72">
        <div>
  {/* Top-level section title */}
  <h2 className="text-2xl font-bold text-white">KNN Algorithm</h2>
  <ul className="list-disc list-inside text-lg text-muted-foreground mt-2">
    <li>Store labeled training data</li>
    <li>Choose <InlineMath>k</InlineMath> neighbors</li>
    <li>Measure distances to all training points</li>
    <li>Classify based on majority vote</li>
  </ul>
</div>

<div>
  {/* Slightly smaller section title */}
  <h2 className="text-xl font-bold text-white mt-6">Key Formula</h2>
  <div className="mt-2">
    <p className="text-lg font-semibold text-white">
      <span className="font-bold">Distance:</span><br /><br/>
      <InlineMath math="d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}" />
    </p>
  </div>
</div>

<div>
  {/* Even smaller section title but still large */}
  <h2 className="text-lg font-bold text-white mt-6">Applications</h2>
  <ul className="list-disc list-inside text-lg text-muted-foreground mt-2">
    <li>Image recognition</li>
    <li>Recommender systems</li>
    <li>Medical diagnosis</li>
    <li>Spam filtering</li>
    <li>Stock prediction</li>
  </ul>
</div>
        </div>
      </div>

      {/* Sliders */}
      <div className="flex flex-col gap-3 mt-6" style={{ width: 800 }}>
  <div className="flex justify-between items-center">
    <Label className="text-xl">Points</Label>
    <span className="text-lg text-muted-foreground">{config.points}</span>
  </div>
  <Slider
    min={10}
    max={200}
    value={[config.points]}
    onValueChange={([v]) => handleSliderChange("points", v)}
    className="w-full"
  />

  <div className="flex justify-between items-center">
    <Label className="text-xl">Classes</Label>
    <span className="text-lg text-muted-foreground">{config.classes}</span>
  </div>
  <Slider
    min={2}
    max={6}
    value={[config.classes]}
    onValueChange={([v]) => handleSliderChange("classes", v)}
    className="w-full"
  />

  <div className="flex justify-between items-center">
    <Label className="text-xl">K Value</Label>
    <span className="text-lg text-muted-foreground">{config.k}</span>
  </div>
  <Slider
    min={1}
    max={10}
    value={[config.k]}
    onValueChange={([v]) => handleSliderChange("k", v)}
    className="w-full"
  />
</div>


    </div>
  )
}
