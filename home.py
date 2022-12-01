import kivy
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty
from kivy.lang import Builder
from kivy.core.window import Window
#from kivy.uix.image import Ima

# Designate Our .kv design file 
Builder.load_file('layout.kv')

class MyLayout(Widget):
    def select_file(self):
        from plyer import filechooser
        filechooser.open_file(on_selection = self.selected)

    def selected(self, selection):
        if selection:
            print(selection[0])
            self.ids.selected_path.text = selection[0]


class MyApp(App):
	def build(self):
		Window.clearcolor = (1,1,1,1)
		return MyLayout()

if __name__ == '__main__':
	MyApp().run()
