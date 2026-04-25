from api.server import app
import os
import uvicorn


def main():
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))


if __name__ == "__main__":
    main()