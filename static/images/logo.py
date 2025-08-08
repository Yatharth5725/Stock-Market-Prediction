# Optional: Run this Python script to generate a placeholder logo
from PIL import Image, ImageDraw, ImageFont
img = Image.new('RGBA', (100, 100), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)
font = ImageFont.load_default()
draw.text((10, 40), "SP", font=font, fill="black")
img.save("static/images/logo.png")