# MapDisplays

An Endstone plugin that lets you make MapDisplays!!!

MapDisplays are little displays that show what you want *(not yet, though)*

## Setup for Development

1. Clone the repository
```shell
git clone https://github.com/niko-at-chalupa/endstone-mapdisplays && cd endstone-mapdisplays
# Make sure you're in a directory where you work on your projects, not in your ~/ directory!
```

2. Install everything
```shell
python3 -m venv .venv && source .venv/bin/activate
pip install --editable . && maturin develop
```

3. Run endstone
```shell
endstone -i -y
```

...and then you're all set! Please contribute

## WHY Rust???

I don't know either

<img src="readme_resources/mapdisplays_idle.gif" />
<br />


Inspired by the Java mod, WebDisplays.