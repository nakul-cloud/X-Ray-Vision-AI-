@echo off
echo ===================================================
echo       🚀 Starting XRayVision AI...
echo ===================================================
echo.
echo 1. Launching Server Process (Frontend + Backend)...
echo    (Please wait for the backend to initialize ~15s)
echo.

:: Start npm run dev in a new window with a title
start "XRayVision Server" npm run dev

echo 2. Waiting for servers to spin up...
timeout /t 15 /nobreak

echo 3. Opening browser...
start http://localhost:3000

echo.
echo    ✅ Done! The app should be ready.
echo    If you see "Failed to fetch", wait a moment and refresh.
echo.
pause
