
from .basic_ops3 import *
import pdb
__all__ = ['PAM_Module'] # 'CAM_Module', 'semanticModule']
class PAM_Module(object):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim,filters,layer_type='SAME'):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.filters=filters
        self.layer_type=layer_type

        print('in_dim=', in_dim)
        print('in-dim//8=', in_dim // 8)
        if layer_type == 'SAME':
            self.query_conv = Conv3D(in_dim, filters, 1, 1, name="Conv3D_attqs", use_bias=True)
        elif layer_type == 'DOWN':
            self.query_conv = Conv3D(in_dim, filters, 3, 2, name="Conv3D_attqd", use_bias=True)
        elif layer_type == 'UP':
            self.query_conv = Deconv3D(in_dim, filters, 3, 2, out_shape=output_shape, name="Conv3D_attqu", use_bias=True)
         #linear transformation for k
        self.key_conv = Conv3D(in_dim, filters, 1, 1, name="Conv3D_attk", use_bias=True)

        # linear transformation for k
        self.value_conv = Conv3D(in_dim, filters, 1, 1, name="Conv3D_attv", use_bias=True)

        print('Q_conv=',self.query_conv )
        print('K-Conv=',self.key_conv)
        print('V_Conv=',self.value_conv)
        self.gamma = tf.Variable(tf.zeros(1)) #nn.Parameter(tf.zeros(1))

        #self.softmax = tf.nn.softmax(x,dim=-1)
    def forward(self, x):
        m_batchsize, height, width, C = x.size()
        print('x=',x.size())
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 3,2, 1)
        print('proj_query=',proj_query)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        print('proj_key=',proj_key)
        energy = tf.matmul(proj_query, proj_key)
        attention = tf.softmax(energy,dim=-1)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
        print('proj_value=',proj_value)
        out = tf.matmul(proj_value, attention.permute(0, 3,2, 1))
        out = out.view(m_batchsize, height, width,C)
        print('out=',out)
        out = self.gamma * out + x
        print('out=', out)
        return out