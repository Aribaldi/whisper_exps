if not exist .\cuda_dlls.7z (
    powershell -Command Invoke-WebRequest https://github.com/Purfview/whisper-standalone-win/releases/download/libs/cuBLAS.and.cuDNN_CUDA12_win_v1.7z -OutFile .\cuda_dlls.7z
    "C:\Program Files\7-Zip\7z.exe" x *.7z -o".\.pixi\envs\custom"
)
