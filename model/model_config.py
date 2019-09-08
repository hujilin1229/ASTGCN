# -*- coding:utf-8 -*-

import configparser

from mxnet import nd
import numpy as np
import scipy.sparse as sp
from lib.utils import scaled_Laplacian, cheb_polynomial, get_adjacency_matrix


def get_backbones(config_filename, adj_filename, ctx):
    config = configparser.ConfigParser()
    config.read(config_filename)

    K = int(config['Training']['K'])
    num_of_weeks = int(config['Training']['num_of_weeks'])
    num_of_days = int(config['Training']['num_of_days'])
    num_of_hours = int(config['Training']['num_of_hours'])
    num_of_vertices = int(config['Data']['num_of_vertices'])

    adj_mx = get_adjacency_matrix(adj_filename, num_of_vertices)
    L_tilde = scaled_Laplacian(adj_mx)
    cheb_polynomials = [nd.array(i, ctx=ctx)
                        for i in cheb_polynomial(L_tilde, K)]

    backbones1 = [
        {
            "K": K,
            "num_of_chev_filters": 64,
            "num_of_time_filters": 64,
            "time_conv_strides": num_of_weeks,
            "cheb_polynomials": cheb_polynomials
        },
        {
            "K": K,
            "num_of_chev_filters": 64,
            "num_of_time_filters": 64,
            "time_conv_strides": 1,
            "cheb_polynomials": cheb_polynomials
        }
    ]

    backbones2 = [
        {
            "K": K,
            "num_of_chev_filters": 64,
            "num_of_time_filters": 64,
            "time_conv_strides": num_of_days,
            "cheb_polynomials": cheb_polynomials
        },
        {
            "K": K,
            "num_of_chev_filters": 64,
            "num_of_time_filters": 64,
            "time_conv_strides": 1,
            "cheb_polynomials": cheb_polynomials
        }
    ]

    backbones3 = [
        {
            "K": K,
            "num_of_chev_filters": 64,
            "num_of_time_filters": 64,
            "time_conv_strides": num_of_hours,
            "cheb_polynomials": cheb_polynomials
        },
        {
            "K": K,
            "num_of_chev_filters": 64,
            "num_of_time_filters": 64,
            "time_conv_strides": 1,
            "num_of_features": 1,
            "cheb_polynomials": cheb_polynomials
        }
    ]

    all_backbones = [
        backbones1,
        backbones2,
        backbones3
    ]

    return all_backbones


def get_backbones_traffic4cast(config_filename, adj_filename, ctx):
    config = configparser.ConfigParser()
    config.read(config_filename)

    K = int(config['Training']['K'])
    num_of_weeks = int(config['Training']['num_of_weeks'])
    num_of_days = int(config['Training']['num_of_days'])
    num_of_hours = int(config['Training']['num_of_hours'])

    adj_mx = sp.load_npz(adj_filename).toarray()
    # adj_mx = get_adjacency_matrix(adj_filename, num_of_vertices)
    L_tilde = scaled_Laplacian(adj_mx)
    cheb_polynomials = [nd.array(i, ctx=ctx)
                        for i in cheb_polynomial(L_tilde, K)]

    num_filters = 16
    backbones1 = [
        {
            "K": K,
            "num_of_chev_filters": num_filters,
            "num_of_time_filters": num_filters,
            "time_conv_strides": num_of_weeks,
            "cheb_polynomials": cheb_polynomials
        },
        {
            "K": K,
            "num_of_chev_filters": num_filters,
            "num_of_time_filters": num_filters,
            "time_conv_strides": 1,
            "cheb_polynomials": cheb_polynomials
        }
    ]

    backbones2 = [
        {
            "K": K,
            "num_of_chev_filters": num_filters,
            "num_of_time_filters": num_filters,
            "time_conv_strides": num_of_days,
            "cheb_polynomials": cheb_polynomials
        },
        {
            "K": K,
            "num_of_chev_filters": num_filters,
            "num_of_time_filters": num_filters,
            "time_conv_strides": 1,
            "cheb_polynomials": cheb_polynomials
        }
    ]

    backbones3 = [
        {
            "K": K,
            "num_of_chev_filters": num_filters,
            "num_of_time_filters": num_filters,
            "time_conv_strides": num_of_hours,
            "cheb_polynomials": cheb_polynomials
        },
        {
            "K": K,
            "num_of_chev_filters": num_filters,
            "num_of_time_filters": num_filters,
            "time_conv_strides": 1,
            "num_of_features": 3, # for traffic4cast is 3, otherwise 1
            "cheb_polynomials": cheb_polynomials
        }
    ]

    all_backbones = [
        backbones1,
        backbones2,
        backbones3
    ]

    return all_backbones
