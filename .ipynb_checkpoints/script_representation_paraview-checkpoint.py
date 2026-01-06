from paraview.simple import *
data_path = "/home/enarf/Code/hypergeometric/success_points.vts"  # <<-- Change this to your actual file
data = OpenDataFile(data_path)
Show(data)
RenderView1 = GetActiveViewOrCreate('RenderView')

# ----------------------------------------------------------------
# 2️⃣ Outline (White Box with Axis Labels)
# ----------------------------------------------------------------
outline = Outline(Input=data)
outlineDisplay = Show(outline, RenderView1)
outlineDisplay.AmbientColor = [1, 1, 1]
outlineDisplay.DiffuseColor = [1, 1, 1]

# Configure axes grid (supported options only)
RenderView1.AxesGrid = 'GridAxes3DActor'
axes = RenderView1.AxesGrid
axes.Visibility = 1
axes.XTitle = "Lands"
axes.YTitle = "Ramp"
axes.ZTitle = "Bombs"

# Axis tick labels and colors (old API)
axes.GridColor = [1, 1, 1]
axes.XTitleColor = [1, 1, 1]
axes.YTitleColor = [1, 1, 1]
axes.ZTitleColor = [1, 1, 1]
axes.XLabelColor = [1, 1, 1]
axes.YLabelColor = [1, 1, 1]
axes.ZLabelColor = [1, 1, 1]

# Tick format (supported)
axes.XLabelFormat = "%-2.0f"
axes.YLabelFormat = "%-2.0f"
axes.ZLabelFormat = "%-2.0f"

# ----------------------------------------------------------------
# 3️⃣ Volume Representation
# ----------------------------------------------------------------
dataDisplay = Show(data, RenderView1)
ColorBy(dataDisplay, ("POINTS", "success_rate"))
dataDisplay.SetRepresentationType("Volume")

# Use Turbo colormap
colorLUT = GetColorTransferFunction("success_rate")
try:
    colorLUT.ApplyPreset("Turbo", True)
except:
    # Fallback for older ParaView versions
    colorLUT.RGBPoints = [0.0, 0.18995, 0.07176, 0.23217,
                          50.0, 0.20803, 0.7181, 0.47216,
                          100.0, 0.9883, 0.99836, 0.64492]
opacityPWF = GetOpacityTransferFunction("success_rate")
opacityPWF.Points = [0.0, 0.0, 0.5, 0.0, 50.0, 0.1, 0.5, 0.0, 100.0, 1.0, 0.5, 0.0]
opacityPWF.ScalarRangeInitialized = 1

# ----------------------------------------------------------------
# 4️⃣ Isosurface (Contour) at 50%
# ----------------------------------------------------------------
contour = Contour(Input=data)
contour.ContourBy = ["POINTS", "success_rate"]
contour.Isosurfaces = [50.0]
contourDisplay = Show(contour, RenderView1)
contourDisplay.DiffuseColor = [0.95, 0.95, 0.95]
contourDisplay.Opacity = 0.6

# ----------------------------------------------------------------
# 5️⃣ Isovolume for values > 50%
# ----------------------------------------------------------------
threshold = Threshold(Input=data)
threshold.Scalars = ["POINTS", "success_rate"]
threshold.ThresholdRange = [50.0, 100.0]
thresholdDisplay = Show(threshold, RenderView1)
thresholdDisplay.SetRepresentationType("Surface")
thresholdDisplay.Opacity = 0.4
thresholdDisplay.DiffuseColor = [0.1, 0.7, 0.2]

# ----------------------------------------------------------------
# 6️⃣ Camera and View Settings (Isometric)
# ----------------------------------------------------------------
RenderView1.OrientationAxesVisibility = 1
RenderView1.Background = [0.05, 0.05, 0.05]
RenderView1.CameraParallelProjection = 1
RenderView1.InteractionMode = '3D'
ResetCamera(RenderView1)
RenderView1.CameraPosition = [1, 1, 1]
RenderView1.CameraViewUp = [0, 0, 1]

# ----------------------------------------------------------------
# 7️⃣ Final Render and Save State
# ----------------------------------------------------------------
RenderAllViews()
print("✅ Visualization pipeline created successfully.")
print("Data file:", data_path)
