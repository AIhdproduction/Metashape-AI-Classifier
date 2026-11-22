# Models

Weights are not included and are ignored by Git. Never commit them. Only the model that you
select is needed.

```text
models/
├── README.md
├── flurseg/
│   └── v1/                         FlurSeg, the default
│       ├── config.json
│       ├── model.safetensors       1,259,264,920 bytes
│       └── preprocessor_config.json
└── sam3.pt                         SAM 3, optional
```

## FlurSeg v1

Run this in the repository folder in PowerShell, after `install.bat`:

```powershell
.venv\Scripts\hf.exe download AIhdproduction/Flurseg --revision 5df53dbbf721778f1ecd4903369e9465926ae2fd --include "v1/*" --local-dir models/flurseg
```

Or download the three files from
[the v1 folder on Hugging Face](https://huggingface.co/AIhdproduction/Flurseg/tree/5df53dbbf721778f1ecd4903369e9465926ae2fd/v1)
and save them, with exactly these names, in `models/flurseg/v1`.

Check the weights. The command must print `True`:

```powershell
(Get-FileHash .\models\flurseg\v1\model.safetensors -Algorithm SHA256).Hash -eq "83ad8990953cadbdbc4ae831eb41eeddf100312709cee8042ff412f211a789a9"
```

The weights are under the DINOv3 License. **Built with DINOv3.**

## SAM 3 (optional, for your own classes)

Request access at https://huggingface.co/facebook/sam3 and accept Meta's SAM License.
Download the SAM 3 image checkpoint and save it here as **sam3.pt**.
