"use client"

import { useEffect, useRef, useState } from "react"
import VisualizationCanvas from "@/components/visualization-canvas"

function useSVMData(params: any) {
  const [points, setPoints] = useState<any[]>([])
  const initialPoints = useRef<any[]>([])
  const prevNoise = useRef<number>(params.noise)

  const generateData = (noise: number, kernel: number) => {
    const numPoints = 15 // Increased points
    const data = []
  
    for (let i = 0; i < numPoints; i++) {
      const x = Math.random() * 4 - 2 // [-2, 2]
      const y = Math.random() * 4 - 2
      let label = 0
  
      if (kernel < 0.5) {
        label = y > 0.5 * x ? (Math.random() > noise ? 1 : 0) : (Math.random() > noise ? 0 : 1)
      } else {
        const dist = Math.sqrt(x * x + y * y)
        label = dist < 1.2 ? (Math.random() > noise ? 1 : 0) : (Math.random() > noise ? 0 : 1)
      }
  
      data.push({ x, y, label })
    }
  
    return data
  }
  

  useEffect(() => {
    if (points.length === 0) {
      const initial = generateData(params.noise, params.kernel)
      initialPoints.current = initial
      setPoints(initial)
    }
  }, [])

  useEffect(() => {
    if (prevNoise.current !== params.noise) {
      prevNoise.current = params.noise
      const newPoints = generateData(params.noise, params.kernel)
      initialPoints.current = newPoints
      setPoints(newPoints)
    } else {
      setPoints(initialPoints.current)
    }
  }, [params.noise, params.kernel])

  return { points, setPoints }
}

function renderSVM(ctx: CanvasRenderingContext2D, width: number, height: number, params: any, points: any[]) {
  const { c, kernel, gamma } = params
  const margin = 40
  const plotWidth = width - 2 * margin
  const plotHeight = height - 2 * margin

  // Update: Adjusted to new margin range [-5, 5]
  const scale = 1.0 // Scale factor for [-5, 5] range
  
  const kernelFunction = (x1: number, y1: number, x2: number, y2: number) => {
    if (kernel < 0.5) return x1 * x2 + y1 * y2
    const dist2 = Math.pow(x1 - x2, 2) + Math.pow(y1 - y2, 2)
    return Math.exp(-gamma * dist2)
  }

  // Simulate C by scaling alpha only for near-boundary points
  const predictRawBase = (x: number, y: number) =>
    points.reduce((sum, point) => {
      const k = kernelFunction(x, y, point.x, point.y)
      return sum + (point.label === 1 ? 1 : -1) * k
    }, 0)

  const marginCutoff = 1 / c
  const alphas = points.map(p => {
    const raw = predictRawBase(p.x, p.y)
    return Math.abs(raw) < marginCutoff ? (p.label === 1 ? c : -c) : 0
  })

  const predictRaw = (x: number, y: number) =>
    points.reduce((sum, point, i) => sum + alphas[i] * kernelFunction(x, y, point.x, point.y), 0)

  const predict = (x: number, y: number) => (predictRaw(x, y) > 0 ? 1 : 0)

  // Clear and draw background
  ctx.clearRect(0, 0, width, height)
  const resolution = 200
  const stepX = plotWidth / resolution
  const stepY = plotHeight / resolution

  for (let px = 0; px < resolution; px++) {
    for (let py = 0; py < resolution; py++) {
      const x = (px / resolution) * 10 - 5  // Updated range to [-5, 5]
      const y = (py / resolution) * 10 - 5  // Updated range to [-5, 5]
      const raw = predictRaw(x, y)

      let color = "rgba(200,200,200,0.1)"
      if (Math.abs(raw) < 1.0) color = "rgba(255,255,0,0.2)"
      else color = raw > 0 ? "rgba(74, 222, 128, 0.2)" : "rgba(248, 113, 113, 0.2)"

      ctx.fillStyle = color
      ctx.fillRect(margin + px * stepX, margin + py * stepY, stepX, stepY)
    }
  }

  // Draw margins for linear kernel
  if (kernel < 0.5) {
    const drawMargin = (level: number, color: string) => {
      ctx.strokeStyle = color
      ctx.beginPath()
      for (let px = 0; px < resolution; px++) {
        const x = (px / resolution) * 10 - 5  // Updated range to [-5, 5]
        let found = false
        for (let py = 0; py < resolution; py++) {
          const y = (py / resolution) * 10 - 5  // Updated range to [-5, 5]
          const raw = predictRaw(x, y)
          if (Math.abs(raw - level) < 0.05) {
            const canvasX = margin + px * stepX
            const canvasY = margin + py * stepY
            if (!found) {
              ctx.moveTo(canvasX, canvasY)
              found = true
            } else {
              ctx.lineTo(canvasX, canvasY)
              break
            }
          }
        }
      }
      ctx.stroke()
    }

    drawMargin(0, "#000")
    drawMargin(1, "#aaa")
    drawMargin(-1, "#aaa")
  }

  // Axes
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

  // Labels
  ctx.fillStyle = "#999"
  ctx.font = "12px sans-serif"
  ctx.textAlign = "center"
  ctx.fillText("-5", margin, height - margin + 20)
  ctx.fillText("0", margin + plotWidth / 2, height - margin + 20)
  ctx.fillText("5", width - margin, height - margin + 20)
  ctx.textAlign = "right"
  ctx.fillText("-5", margin - 10, height - margin)
  ctx.fillText("0", margin - 10, height - margin - plotHeight / 2)
  ctx.fillText("5", margin - 10, margin)

  // Points
  for (const [i, point] of points.entries()) {
    const canvasX = margin + ((point.x + 5) / 10) * plotWidth
    const canvasY = margin + ((point.y + 5) / 10) * plotHeight

    // Highlight support vectors with a border
    const isSupportVector = Math.abs(predictRaw(point.x, point.y)) <= 1.05
    ctx.fillStyle = point.label === 1 ? "#4ade80" : "#f87171"
    ctx.beginPath()
    ctx.arc(canvasX, canvasY, 4, 0, Math.PI * 2)
    ctx.fill()

    if (isSupportVector) {
      ctx.strokeStyle = "#fff"
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.arc(canvasX, canvasY, 6, 0, Math.PI * 2) // Larger circle for support vectors
      ctx.stroke()
    }
  }

  // Info
  ctx.fillStyle = "#fff"
  ctx.font = "14px sans-serif"
  ctx.textAlign = "left"
  ctx.fillText(`C: ${c.toFixed(1)}`, margin + 10, margin + 20)
  ctx.fillText(`Kernel: ${kernel < 0.5 ? "Linear" : "RBF"}`, margin + 10, margin + 40)
  if (kernel >= 0.5) ctx.fillText(`Gamma: ${gamma.toFixed(1)}`, margin + 10, margin + 60)

  const errors = points.filter(p => predict(p.x, p.y) !== p.label).length
  ctx.fillText(`Errors: ${errors}`, margin + 10, margin + (kernel < 0.5 ? 60 : 80))
}


export default function SVMPage() {
  const [params, setParams] = useState({
    c: 1.0,
    kernel: 1,
    gamma: 0.5,
    noise: 0.1,
  })

  const { points } = useSVMData(params)

  const paramControls = [
    { name: "c", label: "C (Regularization)", min: 0.1, max: 2, step: 0.05, defaultValue: 1.0 },
    { name: "kernel", label: "Kernel Type", min: 0, max: 1, step: 1, defaultValue: 1 },
    { name: "gamma", label: "Gamma (RBF Kernel)", min: 0.1, max: 10, step: 0.1, defaultValue: 0.5 },
    { name: "noise", label: "Data Noise", min: 0, max: 0.3, step: 0.05, defaultValue: 0.1 },
  ]

  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-4xl font-bold mb-4">Support Vector Machine</h1>
        <p className="text-lg text-muted-foreground mb-6">
          A supervised learning model that finds the optimal hyperplane separating classes.
        </p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2">
          <VisualizationCanvas
            renderFunction={(ctx, w, h) => renderSVM(ctx, w, h, params, points)}
            params={params}
            setParams={setParams}
            paramControls={paramControls}
            height={500}
          />
        </div>

        <aside className="space-y-6">
          <Section title="How It Works">
            Support Vector Machines find the hyperplane maximizing the margin between classes, determined by support
            vectors.
          </Section>

          <Section title="Kernel Trick">
            <ul className="list-disc list-inside space-y-2">
              <li><strong>Linear:</strong> K(x,y) = x·y</li>
              <li><strong>RBF:</strong> K(x,y) = exp(-γ||x−y||²)</li>
            </ul>
          </Section>

          <Section title="Interactive Controls">
            There a total of 15 points in the dataset.
            <br/><br/>
            <ul className="list-disc list-inside space-y-2">
              <li><strong>C:</strong> Regularization trade-off</li>
              <li><strong>Kernel:</strong> Linear (0) or RBF (1)</li>
              <li><strong>Gamma:</strong> Complexity of RBF boundary</li>
              <li><strong>Noise:</strong> Adds classification challenge</li>
            </ul>
            <br/>
            Note: The points that have a white border are the support vectors.
          </Section>

          <Section title="Applications">
            <ul className="list-disc list-inside space-y-2">
              <li>Text/Image classification</li>
              <li>Handwriting recognition</li>
              <li>Bioinformatics</li>
              <li>Face detection</li>
            </ul>
          </Section>
        </aside>
      </div>
    </div>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h3 className="text-xl font-semibold mb-2">{title}</h3>
      <div className="text-muted-foreground">{children}</div>
    </div>
  )
}
