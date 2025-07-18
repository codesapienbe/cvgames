# CVGames Icon Fetcher

## Overview

The CVGames Icon Fetcher is a utility that downloads random images from the Picsum Photos API and saves them as `icon.png` files for all games in the CVGames collection. This ensures that all games have consistent, visually appealing icons without needing to fetch images on-demand during app store startup.

## Features

- **Bulk Download**: Downloads icons for all games at once
- **Smart Skipping**: Skips games that already have icons (unless `--force` is used)
- **Error Handling**: Graceful handling of network errors and timeouts
- **Progress Tracking**: Real-time progress updates and summary statistics
- **Respectful API Usage**: Includes delays between requests to be respectful to the API
- **Configurable**: Supports different icon sizes and force update mode

## Usage

### Basic Usage
```bash
cd src/cvstore
python fetch_game_icons.py
```

### Force Update All Icons
```bash
python fetch_game_icons.py --force
```

### Custom Icon Size
```bash
python fetch_game_icons.py --size 256
```

### Combined Options
```bash
python fetch_game_icons.py --force --size 192
```

## How It Works

1. **Discovery**: Scans the `src/` directory for all game modules
2. **Validation**: Checks if each directory contains a main game file (`__init__.py`, `main.py`, etc.)
3. **Icon Check**: Verifies if `icon.png` already exists in each game directory
4. **Download**: Fetches random images from Picsum Photos API (https://picsum.photos)
5. **Storage**: Saves images as `icon.png` in each game's directory
6. **Summary**: Provides detailed statistics about the operation

## API Details

- **Source**: Picsum Photos (https://picsum.photos)
- **Format**: PNG images
- **Default Size**: 128x128 pixels
- **Rate Limiting**: 0.5 second delay between requests
- **Timeout**: 10 seconds per request

## File Structure

After running the fetcher, each game directory will contain:
```
src/
├── game_name/
│   ├── __init__.py
│   ├── README.md
│   ├── PLAYER.md
│   ├── icon.png          # ← Newly added
│   └── Resources/        # (if any)
```

## Integration with App Store

The app store has been updated to:
1. **Use Local Icons**: Load `icon.png` files directly from game directories
2. **Fallback Handling**: Show appropriate messages when icons are missing
3. **Performance**: No more on-demand API calls during app store startup

## Benefits

### Performance
- **Faster Startup**: No API calls during app store initialization
- **Consistent Loading**: All icons load instantly from local storage
- **Reduced Network Usage**: One-time download instead of repeated requests

### User Experience
- **Visual Consistency**: All games have proper icons
- **Professional Appearance**: Clean, modern app store interface
- **No Loading Delays**: Icons appear immediately

### Development
- **Easy Management**: All icons in one place
- **Version Control**: Icons can be committed to git
- **Customization**: Easy to replace individual icons manually

## Troubleshooting

### Common Issues

1. **Network Errors**
   - Check internet connection
   - Verify Picsum Photos API is accessible
   - Try running again (temporary network issues)

2. **Permission Errors**
   - Ensure write permissions to game directories
   - Check disk space availability

3. **Missing Icons**
   - Run with `--force` to re-download all icons
   - Check individual game directories for `icon.png` files

### Manual Icon Replacement

To replace a specific game's icon:
1. Delete the existing `icon.png` file
2. Add your custom `icon.png` file
3. Ensure the new icon is 128x128 pixels (or your preferred size)

## Statistics

Based on the latest run:
- **Total Games**: 82
- **Success Rate**: 100%
- **Icons Downloaded**: 82
- **Failed Downloads**: 0

## Future Enhancements

Potential improvements for the icon fetcher:
- **Category-based Icons**: Fetch icons based on game categories
- **Custom Icon Sources**: Support for multiple image APIs
- **Icon Validation**: Verify downloaded images are valid
- **Batch Processing**: Process icons in parallel for faster downloads
- **Icon Caching**: Cache downloaded icons to avoid re-downloading

## Dependencies

- `requests`: For HTTP requests to the image API
- `pathlib`: For cross-platform path handling
- `time`: For rate limiting between requests

## License

This utility is part of the CVGames project and follows the same licensing terms. 