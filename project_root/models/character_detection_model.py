import cv2
import numpy as np

class CharacterDetectionModel:
    def __init__(self):
        # EAST text detector
        self.net = cv2.dnn.readNet("frozen_east_text_detection.pb")

    def detect_text_regions(self, image):
        height, width = image.shape[:2]
        new_height = (height // 32) * 32
        new_width = (width // 32) * 32
        ratio_height = height / float(new_height)
        ratio_width = width / float(new_width)

        blob = cv2.dnn.blobFromImage(image, 1.0, (new_width, new_height),
                                     (123.68, 116.78, 103.94), swapRB=True, crop=False)
        self.net.setInput(blob)
        (geometry, scores) = self.net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])

        rectangles = []
        confidences = []

        for y in range(0, geometry.shape[2]):
            scoresData = scores[0][0][y]
            xData0 = geometry[0][0][y]
            xData1 = geometry[0][1][y]
            xData2 = geometry[0][2][y]
            xData3 = geometry[0][3][y]
            anglesData = geometry[0][4][y]

            for x in range(0, geometry.shape[3]):
                if scoresData[x] < 0.5:
                    continue

                offsetX = x * 4.0
                offsetY = y * 4.0
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]

                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)

                rectangles.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])

        boxes = non_max_suppression(np.array(rectangles), probs=confidences)

        results = []
        for (startX, startY, endX, endY) in boxes:
            startX = int(startX * ratio_width)
            startY = int(startY * ratio_height)
            endX = int(endX * ratio_width)
            endY = int(endY * ratio_height)
            results.append((startX, startY, endX, endY))

        return results
