import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F


class deep_base_model2D(nn.Module):
    def __init__(self, Spiral_Img=True, fusion_method='committee'):
        super(deep_base_model2D, self).__init__()

        self.Spiral_Img = Spiral_Img
        self.fusion_method = fusion_method

        if self.Spiral_Img:
            in_features = 4 * 15 * 96
        else:
            in_features = 8 * 8 * 96

        # print('in_features is ', in_features)

        self.block1 = nn.Sequential(nn.Conv2d(1, 24, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(24),
                                    nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(24),
                                    nn.MaxPool2d((2, 2))
                                    # nn.Dropout(0.25)
                                    )
        self.block2 = nn.Sequential(nn.Conv2d(24, 48, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(48),
                                    nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(48),
                                    nn.MaxPool2d((2, 2))
                                    # nn.Dropout(0.25)
                                    )
        self.block3 = nn.Sequential(nn.Conv2d(48, 96, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(96),
                                    nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(96),
                                    nn.MaxPool2d((2, 2))
                                    # nn.Dropout(0.25)
                                    )

        if self.fusion_method == 'committee':
            dense = nn.Sequential(
                nn.Flatten(),
                # nn.Dropout(),
                nn.Linear(in_features=in_features, out_features=512),
                nn.ReLU(),
                nn.Linear(in_features=512, out_features=2),
                # nn.Dropout(),
                # nn.Softmax()
            )
        elif self.fusion_method == 'late' or self.fusion_method == 'mixed':
            dense = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=in_features, out_features=512),
                nn.ReLU(),
                # nn.Dropout(),
            )
        else:
            raise Exception('Fusion method {} is not supported'.format(self.fusion_method))

        self.dense = dense

    def forward(self, input):
        x = self.block1(input)
        x = self.block2(x)
        x = self.block3(x)
        x = self.dense(x)
        return x


class base_model_vgg(nn.Module):
    def __init__(self):
        super(base_model_vgg, self).__init__()
        model = models.vgg16(pretrained=True)
        features = model.features
        self.block1 = []
        self.block2 = []
        self.block3 = []
        self.block4 = []
        self.block5 = []
        for name, module in features.named_children():
            if int(name) <= 4:
                self.block1.append(module)
            elif int(name) <= 9:
                self.block2.append(module)
            elif int(name) <= 16:
                self.block3.append(module)
            elif int(name) <= 23:
                self.block4.append(module)
            elif int(name) <= 30:
                self.block5.append(module)

        self.block1 = nn.Sequential(*self.block1)
        self.block2 = nn.Sequential(*self.block2)
        self.block3 = nn.Sequential(*self.block3)
        self.block4 = nn.Sequential(*self.block4)
        self.block5 = nn.Sequential(*self.block5)
        self.avgpool = model.avgpool
        self.classifier = nn.Flatten()

    def forward(self, im):
        with torch.no_grad():
            x = self.block1(im)
            x = self.block2(x)
            # x = self.block3(x)
        # x = self.block1(im)
        # x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


class SingleView_model(nn.Module):
    def __init__(self, transfer_learning=False, Spiral_Img=False, fusion_method='committee', deep=True):
        super(SingleView_model, self).__init__()

        self.transfer_learning = transfer_learning
        self.Spiral_Img = Spiral_Img
        self.fusion_method = fusion_method
        self.deep = deep

        if not self.transfer_learning:
                self.model = deep_base_model2D(Spiral_Img=self.Spiral_Img, fusion_method=self.fusion_method)
        else:
            model = base_model_vgg()
            self.model = []
            for name, module in model.named_children():
                self.model.append(module)
            self.model.append(nn.Linear(in_features=512 * 7 * 7, out_features=512))
            self.model.append(nn.ReLU())
            self.model.append(nn.Linear(in_features=512, out_features=2))
            self.model = nn.Sequential(*self.model)
            # print(self.model)
            # print('0')

    def forward(self, im):
        x = self.model(im)
        return x


class MultiView_model(nn.Module):
    def __init__(self, transfer_learning=False, Spiral_Img=False, fusion_method='committee', deep=True):
        super(MultiView_model, self).__init__()

        self.transfer_learning = transfer_learning
        self.Spiral_Img = Spiral_Img
        self.fusion_method = fusion_method
        in_features = 512
        if not transfer_learning:
            self.model1 = deep_base_model2D(Spiral_Img=Spiral_Img, fusion_method=fusion_method)
            self.model2 = deep_base_model2D(Spiral_Img=Spiral_Img, fusion_method=fusion_method)
            self.model3 = deep_base_model2D(Spiral_Img=Spiral_Img, fusion_method=fusion_method)
            self.model4 = deep_base_model2D(Spiral_Img=Spiral_Img, fusion_method=fusion_method)
            self.model5 = deep_base_model2D(Spiral_Img=Spiral_Img, fusion_method=fusion_method)
            self.model6 = deep_base_model2D(Spiral_Img=Spiral_Img, fusion_method=fusion_method)
            self.model7 = deep_base_model2D(Spiral_Img=Spiral_Img, fusion_method=fusion_method)
            self.model8 = deep_base_model2D(Spiral_Img=Spiral_Img, fusion_method=fusion_method)
            self.model9 = deep_base_model2D(Spiral_Img=Spiral_Img, fusion_method=fusion_method)
        else:
            model_ = base_model_vgg()
            model = []
            for name, module in model_.named_children():
                model.append(module)
            model.append(nn.Linear(in_features=in_features * 7 * 7, out_features=in_features))
            model.append(nn.ReLU())
            model = nn.Sequential(*model)
            self.model1 = model
            self.model2 = model
            self.model3 = model
            self.model4 = model
            self.model5 = model
            self.model6 = model
            self.model7 = model
            self.model8 = model
            self.model9 = model
            # model = models.vgg16(pretrained=True)
            # print(model)

        self.classifer = nn.Sequential(
            nn.Linear(in_features=in_features * 9, out_features=2)
        )


    def forward(self, im):

        B, N, C, H, W = im.shape

        view1 = self.model1(im[:, :, 0, :, :])
        view2 = self.model2(im[:, :, 1, :, :])
        view3 = self.model3(im[:, :, 2, :, :])
        view4 = self.model4(im[:, :, 3, :, :])
        view5 = self.model5(im[:, :, 4, :, :])
        view6 = self.model6(im[:, :, 5, :, :])
        view7 = self.model7(im[:, :, 6, :, :])
        view8 = self.model8(im[:, :, 7, :, :])
        view9 = self.model9(im[:, :, 8, :, :])
        x = torch.cat((view1, view2, view3, view4, view5, view6, view7, view8, view9), 1)
        x = self.classifer(x)


        return x


class Spiral_3channels_model(nn.Module):
    def __init__(self, transfer_learning=False, Spiral_Img=True, fusion_method='committee', deep=True):
        super(Spiral_3channels_model, self).__init__()

        self.transfer_learning = transfer_learning
        self.Spiral_Img = Spiral_Img
        self.fusion_method = fusion_method
        in_features = 512
        if not transfer_learning:
            self.model1 = deep_base_model2D(Spiral_Img=Spiral_Img, fusion_method=fusion_method)
            self.model2 = deep_base_model2D(Spiral_Img=Spiral_Img, fusion_method=fusion_method)
            self.model3 = deep_base_model2D(Spiral_Img=Spiral_Img, fusion_method=fusion_method)
        else:
            model = base_model_vgg()
            self.model = []
            for name, module in model.named_children():
                self.model.append(module)
            self.model.append(nn.Linear(in_features=in_features * 7 * 7, out_features=in_features))
            self.model.append(nn.ReLU())
            self.model1 = nn.Sequential(*self.model)
            self.model2 = nn.Sequential(*self.model)
            self.model3 = nn.Sequential(*self.model)
            # model = models.vgg16(pretrained=True)
            # print(model)


        self.classifer = nn.Sequential(
            nn.Linear(in_features=in_features * 3, out_features=2)
        )


    def forward(self, im):
        B, N, C, H, W = im.shape
        # print('B, N, C, H, W', B, N, C, H, W)
        view1 = self.model1(im[:, :, 0, :, :])
        view2 = self.model2(im[:, :, 1, :, :])
        view3 = self.model3(im[:, :, 2, :, :])
        x = torch.cat((view1, view2, view3), 1)
        x = self.classifer(x)
        return x


# if __name__ == '__main__':
#
#     input = torch.rand(64,1,64,64)
#     mod = base_model_vgg()
#     print(mod)
#
#     print('dddd')

