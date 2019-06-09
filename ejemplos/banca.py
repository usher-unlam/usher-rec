#Modulo para el manejo de bancas

import wx
import numpy as np

class Banca():
  def __init__(self,stBitmap,x,y,nro):
    #StaticBitmap(que contiene imagen),xmin,ymin,nroBanca,estado,seleccionado
    self.staticBitmap=stBitmap #Contendra la imagen
    self.xminXML=x #coordenada x left XML
    self.yminXML=y #coordenada y top XML
    self.xminVentana=0 #coordenada x left dentro de la ventana
    self.yminVentana=0 #coordenada y top dentro de la ventana
    self.nro=nro #nro de banca
    self.estado="libre" #estado: libre,ocupada,indeterminado
    self.seleccionado=False #seleccionado:True,False
    self.tamImag=50

  def setImagen(self,path):
    bitmap = wx.Bitmap(path)
    path = escalar_bitmap(bitmap, self.tamImag, self.tamImag)
    self.staticBitmap.SetBitmap(wx.Bitmap(path))

  def setPosicionVentana(self,x,y):
    self.xminVentana=x #coordenada x left dentro de la ventana
    self.yminVentana=y #coordenada y top dentro de la ventana
    self.staticBitmap.SetPosition(wx.Point(x,y))

  def setSeleccionado(self,seleccionado):
    self.seleccionado=seleccionado

  def setEstado(self,estado):
        self.estado=estado
  
  def getEstado(self):
        return self.estado

  def getPosicionXML(self):
    return self.xminXML,self.yminXML

  def getNroBanca(self):
        return self.nro

  def getSeleccionado(self):
       return self.seleccionado

  def getStaticBitmap(self):
       return self.staticBitmap

def escalar_bitmap(bitmap, width, height):
    image = wx.ImageFromBitmap(bitmap)
    image = image.Scale(width, height, wx.IMAGE_QUALITY_HIGH)
    result = wx.BitmapFromImage(image)
    return result

#   
#   
# #calculate distance from another point
# def get_distance(self, other_rocket):
#  
#   return distance  