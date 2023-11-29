from __future__ import print_function
"""
from .lrxp import LRExplainer
from .train_global_model import load_change_metrics_df
from .options import Options
from .rndmforest import XRF, Dataset
from .html_string import HtmlString
"""
from lrxp import LRExplainer
from train_global_model import load_change_metrics_df
from options import Options
from rndmforest import XRF, Dataset
from html_string import HtmlString

import random
import pandas as pd
import ipywidgets as widgets


class FxExplainer(object):
    """A FxExplainer object should be initialized with the following attributes to perform the logical explanation

    Parameters
    ----------
    global_model_name : :obj:`str`
        The name of the black-box global model, currently support 2 models, 'LR' for Logistic Regression and 'RF' for Random Forest
    xtype : :obj:`str`
        Explanation type, currently support 2 types, 'abd' for Abductive Explanation and 'con' for Concretive Explanation
    xnum : :obj:`int`
        Number of explanations to be generated for each instance
    global_model_path : :obj:`str`
        Path to the global model file in .pkl format trained by the sklearn library
    proj_name : :obj:`str`
        Project name
    data_path : :obj:`str`
        Path to the data files required for the FxExplainer
    """

    def __init__(self, global_model_name, xtype, xnum, global_model_path, proj_name, data_path):
        self.options = Options(global_model_name=global_model_name, 
                               xtype=xtype, 
                               xnum=xnum,
                               global_model_path=global_model_path,
                               proj_name=proj_name,
                               data_path=data_path)
        self.explainer = None
        self.gui = None
    
    def prepare_widgets(self, explanation_html, explained_instance_html, instance_id=0) -> None:
        # set up one Accordion for each instance
        tab_nest = widgets.Tab()
        accordion = widgets.Accordion(children=[tab_nest])
        accordion.set_title(index=0, title=[f"Instance ID {instance_id}"])
        abd_con_exp_html = widgets.HTML(value=explanation_html)
        instance_info_html = widgets.HTML(value=explained_instance_html)

        tab_nest.children = [abd_con_exp_html, instance_info_html]
        exp_title = "Abductive Exp." if self.options.xtype == "abd" else "Contrastive Exp."
        tab_nest.set_title(index=0, title=exp_title)
        tab_nest.set_title(index=1, title="Explained Instance")
        self.gui = accordion

    def show_in_jupyter(self) -> None:
        from IPython.display import display
        return display(self.gui)
    
    def explain(self, in_jupyter=False):
        """ Main function to perform the logical explanation
    
        """
        if in_jupyter:
            self.options.in_jupyter = True
        options = self.options
        # explaining
        if options.xtype:
            print('\nExplaining the {0} model...\n'.format('logistic regression' if options.global_model_name == 'LR' else 'random forest'))
            # Explain data
            change_metrics, bug_label = load_change_metrics_df(options.proj_name, options)

            with open(options.data_path + options.proj_name + '_non_correlated_metrics.txt', 'r') as f:
                metrics = f.read()

            metrics_list = metrics.split('\n')
            non_correlated_change_metrics = change_metrics[metrics_list]

            non_correlated_change_metrics['defect'] = bug_label

            non_correlated_change_metrics.to_csv(options.data_path+options.proj_name+'.csv', index=False)

            data = Dataset(filename=options.data_path+options.proj_name+'.csv', mapfile=options.mapfile,
                        separator=options.separator, use_categorical=options.use_categorical)

            insts = pd.read_csv(options.data_path + options.proj_name + '_X_test.csv')

            if len(insts) > 100:
                random.seed(1000)
                selected_ids = random.sample(range(len(insts)), 1)

            nof_inst = 0
            for id in range(len(insts)):
                if id not in selected_ids:
                    continue
                nof_inst += 1
                inst = insts.iloc[id]
                # explain RF model
                if options.global_model_name == 'RF':
                    self.explainer = XRF(data, options)
                # explain LR model
                elif options.global_model_name == 'LR':
                    self.explainer = LRExplainer(data, options)
                
                _, _, explained_instance, explanation, explanation_size = self.explainer.explain(inst)

                if in_jupyter:
                    explained_instance = self.exp_mapping(explained_instance)
                    explanation = self.exp_mapping(explanation)
                    explained_instance_html = HtmlString(list_of_pair=explained_instance, exp_type=self.options.xtype, is_explained_instance=True).get_html()
                    explanation_html = HtmlString(list_of_pair=explanation, exp_type=self.options.xtype).get_html()
                    self.prepare_widgets(explanation_html=explanation_html, explained_instance_html=explained_instance_html)
                    self.show_in_jupyter()
                else:
                    exp_type_name = "Abductive" if self.options.xtype == "abd" else "Contrastive"
                    print("Explained Instance\n", explained_instance, f"\n\n{exp_type_name} Explanation\n", explanation, "\n\n", explanation_size)

    def exp_mapping(self, if_else_text):
        # use list to preserve the order of the if-else statements
        mapped = []
        # map features
        feature_value = if_else_text.split('THEN')[0]
        feature_value = feature_value.split('AND')
        feature_value = [word.strip("IF ") for word in feature_value]
        for fea_val_pair in feature_value:
            fea_val = fea_val_pair.split('=')
            mapped.append([fea_val[0].strip(), round(float(fea_val[1].strip()), 5)])
        # map label
        label_value = if_else_text.split('THEN')[1].strip().split("=")
        mapped.append([label_value[0].strip(), label_value[1].strip()])
        return mapped

'''
fx = FxExplainer(global_model_name="LR", 
                     xtype="abd", 
                     xnum=1, 
                     global_model_path="./global_model/openstack_LR_global_model.pkl", 
                     proj_name="openstack", 
                     data_path="./dataset/")
fx.options.validate = True
fx.explain()
'''