%% --------- Gender-considered Evaluation --------- 
% This script is used to assess the gender influence on factor decomposition.
% The evaluation pipleine has been tested in MatLab 2021a and later.

% The gender influence on models of interest (e.g., 2F and 5F models) are evaluated by addressing two questions. 

% Question 1: Whether the overall model fits better to certain gender groups? 
% To this end, we applied the overall model, which was built on the whole data, to groups with different gender labels, and calculated the 
% reconfiguration errors amongst them. In addition, we randomly replaced the gender labels for several times (e.g., n = 100), and repeated the 
% same aforementioned procedures. Then real value of reconfiguration error was compared to their random values by one-sample T test. If the 
% "real-random" differences are significantly lower than zero, the overall model can be considered to better explain the variation in a given group.

% Question 2: Does a specific gender group exhibit greater intra-group heterogeneity? 
% To this end, we separately built models for different gender groups (e.g., female-specific model and adult-specific model). Then, the reconfiguration 
% errors based on gender-specific models were calculated. Likewise, certain number of random models (e.g., n = 100) were generated by randomly replacing 
% the gender or age labels and repeating the same procedures. Since models were constructed for specific subgroups themselves, a lower reconstruction error 
% relative to that of the random models indicates less intra-group heterogeneity compared to randomly selected subgroups of equal size (one-sample t-test).

%% Toolkit configuration

% Set up path for core codes.

clc; clear;  
base_dir = pwd; % root directory

addpath(genpath(fullfile(base_dir,'Code')));

%% Data loading

% Load demo data.
% "data" is a subjects (rows) * items (colums) matrix.
% "sex" and "age" are corresponding demographic variables.

load(fullfile(base_dir,'Data','data_evaluation.mat')); 

%% Score setting

% Recode reverse scoring items (range: 0-4).
% "reverse" marks items that are scored in reverse.
% Items that require reverse scoring are marked as 1 and other items are marked as 0.

f = find(reverse==1); % reverse-scored items
data(:,f) = max(max(data)) - data(:,f); % turn all items into the same direction

%% Influences of gender 

clear DM model CV OS;
DM.group = sex; % identify gender labels (0:female; 1:male)
DM.group_label = {'Female','Male'}; % set character labels for both groups
DM.class = 'cat'; % models would be separately built in different categories (i.e., female and male)
DM.permutation = 100; % randomly disorganize the gender labels 100 times, and generate the models based on random category labels
DM.missing = 'mode';

model.fmatrix = 'OPNMF'; 
model.factor = [2:8];

CV = [];

OS.filename = 'test';

[Model,EVA,Log,Data] = fmatrix(data,model,DM,CV,OS); % generate group-level models 
 
%% Result outputs & visualization

parameter.factor = [2 5]; % specify the number of factors to display.
parameter.category = 'Gender'; % specify the category label
parameter.color = [0 0 1; 1 0 0]; % specify the RGB color corresponding to Question 1 (blue) and Question 2 (red)

[out,question1,question2] = plot_permutation(Model,EVA,Log,Data,parameter); % result output and display

% "out" contains the statistical parameters suggested for the two questions using the t-test.

% "question1" involves real Reconstruction Error (real gender labels and a single model built on entire sample), 
% random Reconstruction Error (random gender labels and a single model built on entire sample), 
% and their difference values (for both 2F and 5F models, as well as, for both female and male groups).
% The "question1" essentially reflects the generalization effect of applying the overall model to different subgroups.

% "question2" involves real Reconstruction Error (real gender labels and real gender-specific models), 
% random Reconstruction Error (random gender labels and random gender-specific models), 
% and their difference values (for both 2F and 5F models, as well as, for both female and male groups).
% The "question2" essentially reflects the intra-group heterogeneity when applying the group-specific models to their own subgroups.