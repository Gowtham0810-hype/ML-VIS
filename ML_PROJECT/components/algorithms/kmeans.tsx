"use client"

import React, { useEffect, useRef, useState } from "react"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import "katex/dist/katex.min.css"
import { InlineMath } from "react-katex"

const colors = ["red", "green", "blue", "orange", "purple", "cyan"]

export default function KMeansVisualizer() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animRef = useRef<number | null>(null)

  const [config, setConfig] = useState({
    points: 100,
    clusters: 1,
    iterations: 10,
  })
  const [version, setVersion] = useState(0)
  const [data, setData] = useState<{ x: number; y: number; cluster?: number }[]>([])
  const [centroids, setCentroids] = useState<{ x: number; y: number }[]>([])

  // Generate random points
  useEffect(() => {
    const newData = Array.from({ length: config.points }, () => ({
      x: Math.random() * 780 + 10,
      y: Math.random() * 580 + 10,
    }))
    setData(newData)
  }, [config.points])

  // Generate initial centroids
  useEffect(() => {
    const newCentroids = Array.from({ length: config.clusters }, () => ({
      x: Math.random() * 780 + 10,
      y: Math.random() * 580 + 10,
    }))
    setCentroids(newCentroids)
  }, [config.clusters])

  // K-Means animation logic
  useEffect(() => {
    if (!data.length || !centroids.length) return
  
    let points = [...data]
    let centers = [...centroids]
    let iter = 0
  
    const step = () => {
      points = points.map(p => ({ ...p, cluster: undefined }))
      points = points.map((p) => {
        let minDist = Infinity
        let cluster = 0
        centers.forEach((c, i) => {
          const dist = (p.x - c.x) ** 2 + (p.y - c.y) ** 2
          if (dist < minDist) {
            minDist = dist
            cluster = i
          }
        })
        return { ...p, cluster }
      })
  
      const sums = Array.from({ length: config.clusters }, () => ({ x: 0, y: 0, count: 0 }))
      points.forEach((p) => {
        const cluster = p.cluster ?? 0
        if (sums[cluster]) {
          sums[cluster].x += p.x
          sums[cluster].y += p.y
          sums[cluster].count++
        }
      })
      
  
      centers = sums.map((s) => ({
        x: s.count ? s.x / s.count : Math.random() * 780 + 10,
        y: s.count ? s.y / s.count : Math.random() * 580 + 10,
      }))
  
      draw(points, centers)
  
      if (++iter < config.iterations) {
        animRef.current = requestAnimationFrame(step)
      } else {
        setData(points)
        setCentroids(centers)
      }
    }
  
    step()
    return () => cancelAnimationFrame(animRef.current!)
  }, [version, data, centroids])
  
  

  const draw = (points: any[], centers: any[]) => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext("2d")
    if (!canvas || !ctx) return
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    const groups: Record<number, { x: number; y: number }[]> = {}
    for (const p of points) {
      const cluster = p.cluster ?? 0
      if (!groups[cluster]) groups[cluster] = []
      groups[cluster].push(p)
    }

    Object.entries(groups).forEach(([clusterId, pts]) => {
      const cluster = parseInt(clusterId)
      ctx.beginPath()
      for (const p of pts) {
        ctx.moveTo(p.x + 5, p.y)
        ctx.arc(p.x, p.y, 8, 0, 2 * Math.PI)
      }
      ctx.fill()
    })
    

    for (const p of points) {
      ctx.beginPath()
      ctx.arc(p.x, p.y, 4, 0, 2 * Math.PI)
      ctx.fillStyle = colors[p.cluster ?? 0]
      ctx.fill()
    }

    for (const [i, c] of centers.entries()) {
      ctx.beginPath()
      ctx.arc(c.x, c.y, 8, 0, 2 * Math.PI)
      ctx.strokeStyle = colors[i]
      ctx.lineWidth = 3
      ctx.stroke()

      const clusterSize = groups[i]?.length ?? 0
      ctx.fillStyle = "#ffffff"
      ctx.font = "14px sans-serif"
      ctx.fillText(`Cluster ${i + 1} (${clusterSize})`, c.x + 10, c.y - 10)
    }
  }

  const handleSliderChange = (key: keyof typeof config, value: number) => {
    setConfig((c) => ({ ...c, [key]: value }))
    setVersion((v) => v + 1)
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="text-2xl font-bold">K-Means Visualizer</div>
      <div className="text-muted-foreground text-[16px]">
        Watch the K-Means algorithm cluster random data points into groups.
      </div>

      <div className="flex flex-col lg:flex-row gap-6">
        <div className="flex flex-col gap-6">
          <canvas
            ref={canvasRef}
            width={800}
            height={600}
            className="rounded border shadow"
          />

          <div className="flex flex-col gap-4">
            {[
              { label: "Points", key: "points", min: 10, max: 300 },
              { label: "Clusters", key: "clusters", min: 1, max: 6 },
            ].map(({ label, key, min, max }) => (
              <div key={key} className="flex flex-col">
                <div className="flex justify-between items-center">
                  <Label>{label}</Label>
                  <br/>
                  <span className="text-sm text-muted-foreground">
                    {config[key as keyof typeof config]}
                  </span>
                </div>
                <Slider
                  min={min}
                  max={max}
                  value={[config[key as keyof typeof config] as number]}
                  onValueChange={([val]) =>
                    handleSliderChange(key as keyof typeof config, val)
                  }
                />
              </div>
            ))}
          </div>
          
        </div>

        <div className="text-[16px] text-muted-foreground space-y-6 max-w-md">
          <div>
            <div className="text-lg font-semibold text-white mb-1">K-Means Overview</div>
            <ul className="list-disc list-inside space-y-1">
              <li>Initialize random centroids</li>
              <li>Assign points to the nearest centroid</li>
              <li>Recalculate centroids</li>
              <li>Repeat until convergence or max iterations</li>
            </ul>
          </div>

          <div>
            <div className="text-lg font-semibold text-white mb-1">Key Formulas</div>
            <div className="mb-2">
              <span className="text-white font-semibold">Distance:</span><br />
              <br/>
              <p className="text-white font-semibold"><InlineMath math="d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}" /></p>
            </div>
            <br/>
            <div>
              <span className="text-white font-semibold">Centroid:</span><br/>
              <br/>
              <p className="text-white font-semibold"><InlineMath math="c_i = \frac{1}{n} \sum_{j=1}^{n} x_j" /></p>
            </div>
          </div>

          <div>
            <div className="text-lg font-semibold text-white mb-1">Applications</div>
            <ul className="list-disc list-inside space-y-1">
              <li>Customer segmentation</li>
              <li>Image compression</li>
              <li>Document clustering</li>
              <li>Anomaly detection</li>
              <li>Product grouping</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}
