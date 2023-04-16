# Image Classification

Hiermit können Bilder durch verschiedenen Pytorch Modellen klassifiziert und die Wahrscheinlichkeiten ausgegeben werden.

Im Code muss aktuell noch manuell angepasst werden, welches Bild klassifiziert werden soll, dies geschieht über folgende Codezeile:
```
input_image = Image.open("000002_1.jpg")
```

Die Datei *imagenet_classes.txt* behinhaltet die Klassen, zu welchen ein Bild zugeordnet werden kann und stammt von der [Pytorch Seite](https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt)