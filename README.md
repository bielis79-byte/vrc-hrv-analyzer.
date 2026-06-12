# VRC / HRV RRi Analyzer Pro v3.3

Incluye:
- Segmentación por ventanas.
- HRV lineal, frecuencial y no lineal.
- PSD con interpolación cúbica, 4 Hz, Welch 256 s y 50% overlap.
- Smoothness Priors λ=500 para PSD, ApEn, SampEn y MSE.
- RQA aproximada tipo Kubios: m=10, tau=1, r=sqrt(m)*SD.
- HVG real con clasificación orientativa Small World / Scale-Free / Intermedia.
- Diagrama de Poincaré por fase.
- Exportación Excel, CSV, PNG y ZIP.

Ejecutar:

```cmd
streamlit run app.py
```
