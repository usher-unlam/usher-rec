#Modulo para el manejo de bancas

import wx
import numpy as np

class Banca():
  def __init__(self,stBitmap,xmin,ymin,xmax,ymax,nro):
    #StaticBitmap(que contiene imagen),xmin,ymin,nroBanca,estado,seleccionado
    self.staticBitmap=stBitmap #Contendra la imagen
    self.xminXML=xmin #coordenada x left XML
    self.yminXML=ymin #coordenada y top XML
    self.xmaxXML=xmax #coordenada x left XML + Widht
    self.ymaxXML=ymax #coordenada y top XML + Height
    self.xminVentana=0 #coordenada x left dentro de la ventana
    self.yminVentana=0 #coordenada y top dentro de la ventana
    self.nro=nro #nro de banca
    self.estado="libre" #estado: libre,ocupada,indeterminado
    self.seleccionado=False #seleccionado:True,False
    self.mouseEncima=False #mouseEncima:True,False
    self.tamImag=70

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

  def setMouseEncima(self,mouseEncima):
    self.mouseEncima=mouseEncima  

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

  def getMouseEncima(self):
       return self.mouseEncima

  def getStaticBitmap(self):
       return self.staticBitmap

def escalar_bitmap(bitmap, width, height):
    bitmap = bitmap.ConvertToImage() 
    image = wx.Image(bitmap)
    image = image.Scale(width, height, wx.IMAGE_QUALITY_HIGH)
    result = wx.Bitmap(image)
    return result