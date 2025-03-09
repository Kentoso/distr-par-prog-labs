To run on MacOS with OpenMP support, you need to install the OpenMP library. You can do this with Homebrew by running the following command:
```bash
brew install libomp
```

To convert images to png:
```bash
brew install imagemagick
mogrify -format png *.*
```

Example:
```bash
./sobel data30 output 4
```