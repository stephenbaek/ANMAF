import struct

def roiexport(filename, x, y, slice_pos, name):
    """Exports an ImageJ roi file.

    Arguments:
        filename: file path to the roi file
        x: list of x coordinates
        y: list of y coordinates (len(y) == len(x))
        slice_pos: z slice number
        name: name of the roi

    Returns:
        N/A
    """
    with open(filename, 'wb') as file:
        file.write(struct.pack('>ssss', b'I',b'o',b'u',b't'))   # 0-3  Iout
        file.write(struct.pack('>h', 227))                      # 4-5 version
        file.write(bytes([7])) # TYPE = 7  polygon=0; rect=1; oval=2; line=3; freeline=4; polyline=5; noRoi=6; freehand=7; traced=8; angle=9, point=10;
        file.write(bytes([0]))
        top = min(y)
        left = min(x)
        bottom = max(y)
        right = max(x)
        ncoords = len(x)
        file.write(struct.pack('>h', top))   # 8-9
        file.write(struct.pack('>h', left))  # 10-11
        file.write(struct.pack('>h', bottom)) # 12-13
        file.write(struct.pack('>h', right)) # 14-15
        file.write(struct.pack('>h', ncoords)) # 16-17
        file.write(struct.pack('>ffff', 0.0, 0.0, 0.0, 0.0))  # 18-33  X1, Y1, X2, Y2
        file.write(struct.pack('>h', 0))  # 34-35 Stroke width
        file.write(struct.pack('>f', 0.0))  # 36-39 ROI_size
        for i in range(4):
            file.write(bytes([0])) # 40-43 Stroke Color = 0 0 0 0
        for i in range(4):
            file.write(bytes([0])) # 44-47 Fill Color = 0 0 0 0
        file.write(struct.pack('>h', 0)) # 48-49 subtype = 0
        file.write(struct.pack('>h', 0)) # 50-51 options = 0
        file.write(bytes([0])) # 52 arrow stype or aspect ratio = 0
        file.write(bytes([0])) # 53 arrow head size = 0
        file.write(struct.pack('>h', 0)) # 54-55 rounded rect arc size = 0, 0
        file.write(struct.pack('>i', slice_pos))   # 56-59 position = (0, 20)
        h2offset = 4*len(x)+64
        file.write(struct.pack('>i', h2offset))    # 60-63 header2 offset
        for xcoord in x:
            file.write(struct.pack('>h', xcoord - left))
        for ycoord in y:
            file.write(struct.pack('>h', ycoord - top))

        # Header 2
        file.write(bytes([0,0,0,0]))  # 0-3
        file.write(struct.pack('>iii', 0, 0, 0))  # 4-7 C_POSITION, 8-11 Z_POSITION, 12-15 T_POSITION
        file.write(struct.pack('>i', h2offset + 64))   # 16-19 name offset
        file.write(struct.pack('>i', len(name)))   # 20-23 name length
        file.write(struct.pack('>i', 0))   # 24-27 OVERLAY_LABEL_COLOR
        file.write(struct.pack('>h', 0))   # 28-29 OVERLAY_FONT_SIZE

        file.write(bytes([0]))   # 30 GROUP 
        file.write(bytes([0]))   # 31 IMAGE_OPACITY 
        file.write(struct.pack('>i', 0))   # 32-35 IMAGE_SIZE 
        file.write(struct.pack('>f', 0.0))   # 36-39 FLOAT_STROKE_WIDTH 
        file.write(struct.pack('>f', 0.0))   # 40-43 ROI_PROPS_OFFSET  
        file.write(struct.pack('>f', 0.0))   # 44-47 ROI_PROPS_LENGTH   
        file.write(struct.pack('>f', 0.0))   # 48-51 COUNTERS_OFFSET
        for i in range(52, 64):
            file.write(bytes([0]))  # 52-63
        for i in range(len(name)):
            file.write(struct.pack('>h', ord(name[i])))