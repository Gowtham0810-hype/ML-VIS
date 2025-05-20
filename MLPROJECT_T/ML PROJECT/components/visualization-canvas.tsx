import { useRef, useEffect, useState } from "react"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { RefreshCw, Play, Pause } from "lucide-react"

interface VisualizationCanvasProps {
  renderFunction: (
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    params: Record<string, any>,
    frame?: number,
  ) => void
  params: Record<string, any>
  setParams: (params: Record<string, any>) => void
  paramControls: {
    name: string
    label: string
    min: number
    max: number
    step: number
    defaultValue: number
  }[]
  animate?: boolean
  height?: number
  // New optional props for SOM page
  currentFrame?: number
  onAnimateToggle?: () => void
}

export default function VisualizationCanvas({
  renderFunction,
  params,
  setParams,
  paramControls,
  animate = false,
  height = 400,
  currentFrame,
  onAnimateToggle,
}: VisualizationCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  // Only use internal state if currentFrame is not provided externally
  const [internalFrame, setInternalFrame] = useState(0)
  const [isAnimating, setIsAnimating] = useState(false)
  const animationRef = useRef<number | undefined>(undefined)

  // Use either the external frame counter or internal one
  const frame = currentFrame !== undefined ? currentFrame : internalFrame

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    const rect = canvas.getBoundingClientRect()

    canvas.width = rect.width * dpr
    canvas.height = height * dpr

    ctx.scale(dpr, dpr)

    renderFunction(ctx, rect.width, height, params, frame)

    const handleResize = () => {
      const rect = canvas.getBoundingClientRect()
      canvas.width = rect.width * dpr
      canvas.height = height * dpr
      ctx.scale(dpr, dpr)
      renderFunction(ctx, rect.width, height, params, frame)
    }

    window.addEventListener("resize", handleResize)

    return () => {
      window.removeEventListener("resize", handleResize)
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [renderFunction, params, height, frame])

  // Only use internal animation logic if onAnimateToggle is not provided
  useEffect(() => {
    if (!animate || onAnimateToggle) return

    if (isAnimating) {
      const animate = () => {
        setInternalFrame((prev) => prev + 1)
        animationRef.current = requestAnimationFrame(animate)
      }

      animationRef.current = requestAnimationFrame(animate)
    } else if (animationRef.current) {
      cancelAnimationFrame(animationRef.current)
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isAnimating, animate, onAnimateToggle])

  const handleReset = () => {
    const defaultParams = paramControls.reduce(
      (acc, control) => {
        acc[control.name] = control.defaultValue
        return acc
      },
      {} as Record<string, number>,
    )

    setParams(defaultParams)
    
    // Only reset internal frame if we're not using external frame control
    if (currentFrame === undefined) {
      setInternalFrame(0)
    }
  }

  const handleParamChange = (name: string, value: number[]) => {
    setParams({ ...params, [name]: value[0] })
  }

  const handleAnimateToggle = () => {
    if (onAnimateToggle) {
      // Use the external animation control
      onAnimateToggle()
    } else {
      // Use the internal animation control
      setIsAnimating(!isAnimating)
    }
  }

  return (
    <div className="space-y-6">
      <div
        className="canvas-container rounded-lg border overflow-hidden bg-black/20 backdrop-blur-sm"
        style={{ height: `${height}px` }}
      >
        <canvas ref={canvasRef} className="interactive-canvas" style={{ width: "100%", height: `${height}px` }} />
      </div>

      <div className="space-y-4">
        {paramControls.map((control) => (
          <div key={control.name} className="space-y-2">
            <div className="flex justify-between">
              <label className="text-sm font-medium">{control.label}</label>
              <span className="text-sm text-muted-foreground">{params[control.name]}</span>
            </div>
            <Slider
              value={[params[control.name]]}
              min={control.min}
              max={control.max}
              step={control.step}
              onValueChange={(value) => handleParamChange(control.name, value)}
            />
          </div>
        ))}
      </div>

      <div className="flex gap-2">
        <Button variant="outline" size="sm" onClick={handleReset}>
          <RefreshCw className="mr-2 h-4 w-4" />
          Reset
        </Button>

        {animate && (
          <Button variant="outline" size="sm" onClick={handleAnimateToggle}>
            {(onAnimateToggle ? (currentFrame < params.iterations) : isAnimating) ? (
              <>
                <Pause className="mr-2 h-4 w-4" />
                Pause
              </>
            ) : (
              <>
                <Play className="mr-2 h-4 w-4" />
                Animate
              </>
            )}
          </Button>
        )}
      </div>
    </div>
  )
}