from sklearn.base import TransformerMixin



class TemplateRenderer(TransformerMixin):

    def fit(self, raw_text):
        return self

    def decoder(self, prediction):
        if prediction == 0:
            return('The Renaissance (Before 1670)')
        if prediction == 1:
            return('The Restoration and the Age of Enlightenment (1670-1830)')
        if prediction == 2:
            return('The Romantic Period (1830-1890)')
        if prediction == 3:
            return('The Late Victorian Era (Naturalism and Realism) (1890-1920)')
        if prediction == 4:
            return('The Modernist Period (1920-1950)')
        if prediction == 5:
            return('The Contemporary Period (1950-present)')
        else:
            return('Try Again')

    def template_chooser(self, prediction):
        if prediction == 0:
            template = 'result0.html'
        elif prediction == 1:
            template = 'result1.html'
        elif prediction == 2:
            template = 'result2.html'
        elif prediction == 3:
            template = 'result3.html'
        elif prediction == 4:
            template = 'result4.html'
        elif prediction == 5:
            template = 'result5.html'
        else:
            template = 'home.html'

        return template
